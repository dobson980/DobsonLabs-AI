import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZ_OPENAI_API_KEY"),
    base_url=f"https://{os.getenv('AZ_OPENAI_RESOURCE')}.openai.azure.com/openai/v1/",
    default_query={"api-version": "preview"}
)

response = client.responses.create(
    model="gpt4_1_nano",  # This is your deployment name
    input="Im looking for a travel destination for my vacation. I want lots of snow!",
    instructions="You are a helpful travel agent who speaks like Donald Trump."
)


# Print assistant message
assistant_output = response.output[0].content[0].text
print("\nAssistant message:\n", assistant_output)

# Print token usage
print("\nToken usage:")
print("  Input tokens:", response.usage.input_tokens)
print("  Output tokens:", response.usage.output_tokens)
print("  Total tokens:", response.usage.total_tokens)