# EXAMPLE USAGE OF OPENAI API

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
apikey=os.getenv('OPENAI_API_KEY')

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is 1+1?"
        }
    ]
)

print(completion.choices[0].message)