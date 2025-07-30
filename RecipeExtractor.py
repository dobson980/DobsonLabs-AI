import asyncio

# Import Semantic Kernel classes for chat agents and conversation history
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# --- Setup Instructions for Beginners ---
# Before running this script:
# 1. Create a `.env` file in the root of your project with the following keys:
#    AZURE_OPENAI_ENDPOINT=<your Azure OpenAI endpoint>
#    AZURE_OPENAI_API_KEY=<your API key>
#    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
#
# 2. Install dependencies with:
#    pip install -r requirements.txt
#
# Your `requirements.txt` should include at minimum:
# semantic-kernel
# python-dotenv

# --- Define Mini Agents ---

# Agent 1: Extracts recipe content from a given URL
recipe_extraction_agent = ChatCompletionAgent(
    service=AzureChatCompletion(deployment_name='gpt4_1_nano'),  # Uses cost-efficient nano model
    name="RecipeExtractor",
    instructions="""
        You are a recipe extraction assistant. Your task is to extract a structured recipe from the content found at a provided URL.

        Your output should include:
        - A recipe title (if available)
        - A clean, plain-text list of ingredients (one per line)
        - A clean, plain-text list of instructions or steps (one per line)

        Requirements:
        - Do not include ads, comments, promotional text, or unrelated content
        - Remove vague phrases like "as needed"
        - Preserve ingredient and instruction order from the source

        Format output as:

        Title: <Recipe Title>

        Ingredients:
        - <ingredient 1>
        - <ingredient 2>

        Instructions:
        1. <Step one>
        2. <Step two>
    """,
)

# Agent 2: Converts ingredient lines into standardized units and quantities
ingredient_normalization_agent = ChatCompletionAgent(
    service=AzureChatCompletion(deployment_name='gpt4_1_nano'),
    name="IngredientNormalizer",
    instructions="""
        You are an expert ingredient parser. Convert each ingredient line into a clean format using the user's preferred unit system (default is US).

        Format: <amount> <unit> <ingredient>

        - Use numeric decimals (e.g. 1 1/2 -> 1.5)
        - Ignore optional notes like "chopped" or "to taste"
        - Use common US units: cups, tablespoons, ounces, etc.

        Example:
        - 1 cup all-purpose flour
        - 0.5 teaspoon salt
    """,
)

# Agent 3: Generates a real-world shopping list based on normalized ingredients
shopping_list_agent = ChatCompletionAgent(
    service=AzureChatCompletion(deployment_name='gpt4_1_nano'),
    name="ShoppingListGenerator",
    instructions="""
        Generate a shopping list based on the ingredient list provided.

        - Estimate realistic packaging sizes (e.g., 1 lb, 1 dozen)
        - Group similar items
        - No extra commentary

        Format:
        - 1 x 5 lb bag all-purpose flour
        - 1 dozen eggs
    """,
)

# --- Define Orchestrator Agent ---
# This agent handles user input and delegates to the appropriate mini-agents
orchestration_agent = ChatCompletionAgent(
    service=AzureChatCompletion(),  # Uses default deployment from .env (gpt-4o)
    name="Orchestrator",
    instructions="""
        You are an orchestration agent. Your job is to:

        1. Extract recipes from URLs using RecipeExtractor
        2. Normalize ingredients using IngredientNormalizer (US units)
        3. Generate shopping lists if the user asks

        Do not ask questions. Act decisively.

        When given a URL:
        - Extract the recipe
        - Normalize the ingredients
        - Return the full output with:
            Title:
            Ingredients:
            Instructions:

        When the user says something like "shopping list":
        - Generate the shopping list from the normalized ingredients

        Respond only with results. Do not say which agent was used or include metadata.
        If the user says "no" or gives an unrelated query, end politely.
    """,
    plugins=[
        recipe_extraction_agent,
        ingredient_normalization_agent,
        shopping_list_agent
    ]
)

# --- Initialize conversation history thread ---
thread = ChatHistoryAgentThread()

# --- Entry point for the program ---
async def main() -> None:
    print("Welcome to the Recipe Assistant!\n Type 'exit' to quit.")

    while True:
        user_input = input("User:> ")

        if user_input.lower() == "exit":
            print("Exiting the Recipe Assistant. Goodbye!")
            break

        response = await orchestration_agent.get_response(
            messages=user_input,
            thread=thread
        )

        if response:
            print(f"Assistant: {response}")

# Run the async main loop
if __name__ == "__main__":
    asyncio.run(main())