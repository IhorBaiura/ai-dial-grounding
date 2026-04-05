import asyncio
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""

BATCH_SIZE = 100

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens: list[int] = []

    def add_tokens(self, tokens: int) -> None:
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self) -> dict[str, int | list[int]]:
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


token_tracker = TokenTracker()

llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)

def join_context(context: list[dict[str, Any]]) -> str:
    return "\n".join([f"User:\n" + "\n".join([f"  {key}: {value}" for key, value in user.items()]) for user in context])


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response: AIMessage = await llm_client.ainvoke(messages)
    token_usage = response.response_metadata.get('token_usage', {})
    total_tokens = token_usage.get('total_tokens', 0)

    token_tracker.add_tokens(total_tokens)
    
    print(f"Response Content: {response.content}")
    print(f"Total Tokens Used: {total_tokens}")
    
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        user_client = UserClient()
        users = user_client.get_all_users()
        user_batches = [users[i:i + BATCH_SIZE] for i in range(0, len(users), BATCH_SIZE)]
        user_batches = [join_context(batch) for batch in user_batches]
        user_batches = [
            generate_response(
                BATCH_SYSTEM_PROMPT, 
                USER_PROMPT.format(context=context, query=user_question)
            ) for context in user_batches 
        ]

        responses = await asyncio.gather(*user_batches)
        filtered_responses = [resp for resp in responses if resp.strip() != "NO_MATCHES_FOUND"]

        if not filtered_responses:
            print("No users found matching")
        else:
            combined_results = "\n\n".join(filtered_responses)
            final_response = await generate_response(FINAL_SYSTEM_PROMPT, USER_PROMPT.format(context=combined_results, query=user_question))
            print("\n--- Final Response ---")
            print(final_response)

    token_tracker_summary = token_tracker.get_summary()
    print("\n--- Token Usage Summary ---")
    print(f"Total Tokens Used: {token_tracker_summary['total_tokens']}")
    print(f"Number of Batches Processed: {token_tracker_summary['batch_count']}")
    print(f"Tokens Used per Batch: {token_tracker_summary['batch_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())

# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation