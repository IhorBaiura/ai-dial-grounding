import asyncio
from types import CoroutineType
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You should group users by hobby that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment='gpt-4o',
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)


class GroupingResult(BaseModel):
    """One hobby label and the user IDs the model associates with that hobby."""

    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    """Structured LLM output: all hobby groupings for a single user query."""

    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


def format_user_document(user: dict[str, Any]) -> str:
    return f"User:\n  id: {user.get('id')}\n  About user: {user.get('about_me')}\n---"


class InputGrounder:
    CHROMA_COLLECTION_NAME = "users"

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb): # type: ignore
        pass

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all current users."""
        print("🔍 Loading all users for initial vectorstore...")

        user_client = UserClient()
        users = user_client.get_all_users()
        print(f"✅ Loaded {len(users)} users.")

        documents: list[Document] = [Document(id=user.get('id'), page_content=format_user_document(user)) for user in users]

        tasks: list[CoroutineType[Any, Any, list[str]]] = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            if self.vectorstore is None:
                self.vectorstore = Chroma(collection_name=self.CHROMA_COLLECTION_NAME, embedding_function=self.embeddings)
            tasks.append(self.vectorstore.aadd_documents(batch_docs))

        await asyncio.gather(*tasks)

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        """Retrieve context, with optional automatic vectorstore update."""

        await self._update_vectorstore()  # Ensure vectorstore is up-to-date before retrieval

        if not self.vectorstore:
            raise Exception("Vectorstore is not initialized.")

        docs = self.vectorstore.similarity_search_with_relevance_scores(query=query, k=k, score_threshold=score)

        context_parts: list[str] = []
        for doc, relevance_score in docs:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")

        return "\n\n".join(context_parts)

    async def _update_vectorstore(self):
        """Update vectorstore by adding new users and removing deleted ones. 
        This ensures that the vectorstore remains consistent with the User Service."""
        print("🔍 Loading all users for initial vectorstore...")

        if not self.vectorstore:
            raise Exception("Vectorstore is not initialized.")

        user_client = UserClient()
        users = user_client.get_all_users()
        print(f"✅ Loaded {len(users)} users.")

        vectorstore_data: dict[str, Any] = self.vectorstore.get() if self.vectorstore else {"ids": []}
        vectorstore_ids_set = set(str(user_id) for user_id in vectorstore_data.get("ids", []))
        
        users_dict = {str(user.get('id')): user for user in users}
        users_ids_set = set(users_dict.keys())

        new_user_ids = users_ids_set - vectorstore_ids_set
        user_ids_to_delete = vectorstore_ids_set - users_ids_set

        if user_ids_to_delete:
            print(f"Deleting {len(user_ids_to_delete)} users from vectorstore...")
            self.vectorstore.delete(ids=list(user_ids_to_delete))

        new_documents = [Document(id=user_id, page_content=format_user_document(users_dict[user_id])) for user_id in new_user_ids]
        if new_documents:
            print(f"Adding {len(new_documents)} new users to vectorstore...")
            await self.vectorstore.aadd_documents(new_documents)


    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)


    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        parser = PydanticOutputParser(pydantic_object=GroupingResults)

        messages: list[BaseMessage | SystemMessagePromptTemplate] = [
            SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())
        
        grouping_results: GroupingResults = (prompt | self.llm_client | parser).invoke({})
        
        return grouping_results


class OutputGrounder:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        for grouping_result in grouping_results.grouping_results:
            print(f"Hobby: {grouping_result.hobby}")
            print(f"Fetched users: {await self._find_users(grouping_result.user_ids)}\n\n{'='*50}\n")

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                user = await self.user_client.get_user(user_id)
                return user
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise e

        users = await asyncio.gather(*(safe_get_user(user_id) for user_id in ids))
        return [user for user in users if user is not None]


async def main():
    embeddings = AzureOpenAIEmbeddings(
        model='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
        check_embedding_ctx_length=False
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            context = await rag.retrieve_context(user_question)
            if not context:
                print("No relevant information found.")
                continue
            augmented_prompt = rag.augment_prompt(user_question, context)
            grouping_results = rag.generate_answer(augmented_prompt)

            print(f"Grouping Results: {grouping_results}")

            await output_grounder.ground_response(grouping_results)


if __name__ == "__main__":
    asyncio.run(main())
