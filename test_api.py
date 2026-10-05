import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_URL"))

model = os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus")

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Please return API test is successful."}]
)

print(response.choices[0].message.content)