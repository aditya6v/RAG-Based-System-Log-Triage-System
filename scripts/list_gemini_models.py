from dotenv import load_dotenv
load_dotenv()
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for model in client.models.list():
    if "embedContent" in (model.supported_actions or []):
        print(model.name)