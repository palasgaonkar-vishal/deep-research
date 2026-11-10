from agents import OpenAIChatCompletionsModel, set_tracing_export_api_key, function_tool
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os


load_dotenv(override=True)
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))

google_api_key = os.getenv('GOOGLE_API_KEY')
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model="gemini-3.1-flash-lite", openai_client=gemini_client)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
ollama_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
llama3_2_local_model = OpenAIChatCompletionsModel(model="llama3.2", openai_client=ollama_client)
qwen3_5_local_model = OpenAIChatCompletionsModel(model="kimbrasil/qwen3.5:4b", openai_client=ollama_client)

model = llama3_2_local_model if os.getenv("USE_LOCAL_MODEL") == "true" else gemini_model