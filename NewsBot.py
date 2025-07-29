import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=f"https://api.openai.com/v1/",
    default_query={"api-version": "preview"}
)


tools = [
    {
        "type": "web_search_preview",
        "search_context_size": "high"
    }
]

instructions = """
You are a News Aggregation Agent designed to retrieve and summarize same-day news for technology products specified by the user.
Your task is to search only for news published today and return a clean, minimal, itemized list for each product requested. Use only reputable sources (e.g., Microsoft, Apple, major tech publications).

User Input:
The user will specify one or more products or technologies they want news about.

For each product requested, return:
- Recent news headlines
- Security issues or vulnerabilities
- New features or updates

Format:
- Use a bullet-point list grouped by product name.
- Each item should include:
  - A brief one-sentence summary
  - A direct link to the source

Requirements:
- Do not include anything older than today.
- Keep the output clean and minimal—just the list, summaries, and links.
"""

while True:
    user_input = input("Enter a product or technology to get today's news (or type 'exit' to quit): ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting NewsBot. Goodbye!")
        break
    response = client.responses.create(
        model = "gpt-4.1-mini",
        input = user_input,
        tools = tools,
        instructions = instructions
    )
    assistant_output = response.output_text
    print("\nAssistant message:\n", assistant_output)
    print("\nToken usage:")
    print("  Input tokens:", response.usage.input_tokens)
    print("  Output tokens:", response.usage.output_tokens)
    print("  Total tokens:", response.usage.total_tokens)
