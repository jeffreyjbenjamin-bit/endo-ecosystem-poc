import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
)

dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini-dev")

resp = client.chat.completions.create(
    model=dep,
    messages=[{"role": "user", "content": "hello in 3 words"}],
    temperature=0.2,
    max_tokens=20,
)
print(resp.choices[0].message.content)
