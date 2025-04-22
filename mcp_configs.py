from dotenv import load_dotenv
import os

load_dotenv()
mcp_configs = {
    "tavily-mcp": {
        "command": "npx",
        "args": ["-y", "tavily-mcp@0.1.4"],
        "env": {
            "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        },
        "disabled": False,
        "autoApprove": []
    }
}