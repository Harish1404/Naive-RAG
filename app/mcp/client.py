import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.core.config import settings

logger = logging.getLogger(__name__)


# Module-level singleton — managed via connect() / disconnect()
_client: MultiServerMCPClient | None = None
_tools: list = []


def _build_server_config() -> dict:
    """
    Builds the MultiServerMCPClient config for GitHub's remote MCP server.
    Uses streamable_http transport with PAT-based Bearer auth.
    """
    if not settings.GITHUB_PAT:
        raise ValueError("GITHUB_PAT is not set in .env — cannot connect to GitHub MCP server.")

    return {
        "github": {
            "transport": "streamable_http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {
                "Authorization": f"Bearer {settings.GITHUB_PAT}",
            },
        },
        "google-chrome": {
            "transport": "streamable_http",
            "url": "https://mcp.googleapis.com/",
            "headers": {
                "Authorization": f"Bearer {settings.GOOGLE_API_KEY}",
            },
        },
    }


async def connect():
    """Initialize the MCP client and load tools at app startup."""
    global _client, _tools

    try:
        config = _build_server_config()
        _client = MultiServerMCPClient(config)
        _tools = await _client.get_tools()
        logger.info(f"MCP connected — {len(_tools)} tool(s) loaded from GitHub remote server.")
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        _tools = []


async def disconnect():
    """Close the MCP client connection at app shutdown."""
    global _client, _tools

    _client = None
    _tools = []
    logger.info("MCP client disconnected.")



def get_tools() -> list:
    """Return the list of LangChain tools loaded from MCP servers."""
    return _tools
