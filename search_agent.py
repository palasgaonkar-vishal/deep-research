from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, set_tracing_export_api_key, function_tool
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
from ddgs import DDGS
import json
from typing import List, Dict
from model import model

load_dotenv(override=True)


# MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")


@function_tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Searches the live web to fetch real-time information, tutorials, and documentation.
    
    Args:
        query: The search query or question to lookup.
        max_results: The maximum number of search results to return. Defaults to 5.
    """
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(
                query=query,
                region="wt-wt",
                safesearch="moderate",
                max_results=max_results
            )
            
            # Formating data safely using the updated ddgs response format keys
            formatted_results: List[Dict[str, str]] = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", item.get("href", "")),
                    "snippet": item.get("content", item.get("body", ""))
                }
                for item in raw_results
            ]
            return json.dumps({"results": formatted_results}, indent=2)
            
    except Exception as e:
        return json.dumps({"error": f"Failed to execute web search: {str(e)}"})



INSTRUCTIONS = """
You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. Retry the web_search tool if there is an error. Max retries 3. 
The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""

settings = ModelSettings(tool_choice="required")
tools = [web_search]

search_agent = Agent(name="Search Agent", instructions=INSTRUCTIONS, tools=tools, model=model, model_settings=settings)