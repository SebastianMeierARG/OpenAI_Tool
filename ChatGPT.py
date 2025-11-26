from dotenv import load_dotenv
load_dotenv()
import os
# GEWt os enviromet parameters from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
OPENAI_MODEL=os.getenv("OPENAI_MODEL") #we couldn't use gpt-4o-mini
print(OPENAI_MODEL)

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def chat_with_openai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("An error occurred:", e)
        return "Sorry, something went wrong. Please try again."

if __name__ == "__main__":
    while True:
        user_input = input("ASK ANYTHING >> ")
        if user_input.lower() in ["bye", "quit", "exit"]:
            break
        print("RESPONSE:", chat_with_openai(user_input))
