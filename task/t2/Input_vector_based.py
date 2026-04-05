import asyncio
from types import CoroutineType
from typing import Any, Tuple, List
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


SYSTEM_PROMPT = """You are a RAG-powered assistent that assists users with their questions about user information.

## Structure of user message:
`RAG CONTEXT` - Retrieved documents relevant to the user query
`USER QUESTION` - The user's actual question

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`
- Cite specific sources when using information from the context
- Answer ONLY based on conversation history and RAG context
- If no relevant information exists in `RAG CONTEXT` or conversation history state that you cannot answer the question
- Be conversational and helpful in your responses
- When presenting user information, formmat it clearly and include relevant details
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    """Format user information into a string for embedding."""
    return "User:\n" + "\n".join([f"  {key}: {value}" for key, value in user.items()])


class UserRAG:
    INDEX_DIR = "faiss_index"

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore: FAISS | None = None

    async def __aenter__(self):
        print("🔎 Loading all users...")

        user_client = UserClient()
        users = user_client.get_all_users()
        print(f"✅ Loaded {len(users)} users.")

        documents: list[Document] = []
        for user in users:
            page_content = format_user_document(user)
            documents.append(Document(page_content=page_content))
            # print(page_content, "\n---\n")

        self.vectorstore = await self._create_vectorstore_with_batching(documents)

        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100) -> FAISS:
        '''Create a FAISS vector store from documents using batching to handle large datasets.'''
        vector_tasks: list[CoroutineType[Any, Any, FAISS]] = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            vector_tasks.append(FAISS.afrom_documents(batch_docs, self.embeddings))

        vectorstores = await asyncio.gather(*vector_tasks, return_exceptions=True)

        final_vectorstore: FAISS | None = None
        for vectorstore in vectorstores:
            if isinstance(vectorstore, BaseException):
                print(f"Error occurred while creating vectorstore: {vectorstore}")
                continue
            if vectorstore:
                if final_vectorstore is None:
                    final_vectorstore = vectorstore
                else:
                    final_vectorstore.merge_from(vectorstore)

        if final_vectorstore is None:
            raise Exception("Failed to create vectorstore from documents.")
        
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        '''Retrieve relevant context from the vector store based on the user query.'''
        if not self.vectorstore:
            raise Exception("Vectorstore is not initialized.")
        
        context_parts: List[str] = []
        relevant_docs: List[Tuple[Document, float]] = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )
        
        for doc in relevant_docs:
            if doc[1] >= score:
                context_parts.append(doc[0].page_content)
                print(f"Retrieved document with relevance score {doc[1]}:\n{doc[0].page_content}\n---\n")

        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"Augmenting prompt with context...")
        return USER_PROMPT.format(context=context, query=query)


    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"Generating answer for augmented prompt...")
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        response: AIMessage = self.llm_client.invoke(messages)
        return response.content



async def main():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1", 
        dimensions=384, 
        azure_endpoint=DIAL_URL, 
        api_key=SecretStr(API_KEY)
    )

    llm_client = AzureChatOpenAI(
        temperature=0.0,
        azure_deployment="gpt-4o",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version=""
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            context = await rag.retrieve_context(user_question)
            if not context:
                print("No relevant context found. Unable to answer the question.\n")
                continue

            try:
                augmented_prompt = rag.augment_prompt(user_question, context)
                answer = rag.generate_answer(augmented_prompt)
                print(f"Answer:\n{answer}\n\n{'='*50}\n")
            except Exception as e:
                print(f"An error occurred while generating the answer: {e}\n")


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold 
# Benefits are:
#   - Similarity search by context 
#   - Any input can be used for search 
#   - Costs reduce since we don't need to provide all the context for LLM, but only relevant parts