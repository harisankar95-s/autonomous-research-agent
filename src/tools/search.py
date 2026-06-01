from tavily import TavilyClient
from src.tools.base import Tool
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def web_search(query: str) -> str:
    logger.info(f"Searching for: {query}")
    client = TavilyClient(api_key=config.tavily_api_key)
    results = client.search(query)
    contents = [item["content"] for item in results["results"]]
    return "\n\n".join(contents)

web_search_tool = Tool(
    name="web_search",
    description="Search the web for current information on any topic",
    parameters={"query": "the search query string"},
    func=web_search
)

    
