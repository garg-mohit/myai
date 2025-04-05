import requests

API_KEY = ""
API_URL = 'https://api.deepseek.com/v2/generate'  # Example, update based on docs  # Replace with the actual API endpoint


class mydeepseek:

    def __init__(self):
        print(f"{self.__class__.__name__}")

    def generate_text(self, topic):
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            'prompt': f"Write a detailed article about {topic}.",
            'max_tokens': 500,  # Adjust as needed
            'temperature': 0.7  # Adjust for creativity vs. coherence
        }

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            return f"Error: {response.status_code}, {response.text}"

from openai import OpenAI

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
  ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)


# if __name__ == "__main__":
#     myobj = mydeepseek()
#     result = myobj.generate_text("5 benefits of green tea")
#     print(result)