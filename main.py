import logging
from argparse import ArgumentParser

import asyncio
from langchain.agents import create_agent
from dotenv import load_dotenv

from tools import check_for_album, search_for_torrents, add_torrent, KnownTorrents

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="sprintboy.log",
)
logger = logging.getLogger(__name__)


async def main():
    parser = ArgumentParser(description="QBittorrent-Plex-Langchain Demo")
    parser.add_argument("query", type=str, help="Query input for the demo")
    args = parser.parse_args()
    load_dotenv()
    agent = create_agent(
        "anthropic:claude-sonnet-4-5",
        tools=[search_for_torrents, add_torrent, check_for_album],
        context_schema=KnownTorrents,
    )
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can search for torrents using qBittorrent. Based on the user's query, use the tool to find relevant torrents. Select the most appropriate torrents, prefer higher quality, and only choose a vinyl rip if the user specifically requests it. Provide the user with a concise summary of the top results. Then use the add_torrent tool to add the selected torrents to qBittorrent. You can search the existing album database, and you should check to see if you already have an album before recommending it. If there are no albums that satisfy the user query that aren't in the database already, just tell them that and don't try to force it through anyway. You cannot ask additional questions and you should treat ambiguity as a 'dealer's choice type of scenario'",
                },
                {"role": "user", "content": args.query},
            ]
        },
        context=KnownTorrents(torrents={}),
    )
    logger.info(f"Agent response: {response}")
    print("Agent Response:", response)


if __name__ == "__main__":
    asyncio.run(main())
