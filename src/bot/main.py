import asyncio
import logging

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import discord
from discord.ext import commands
from bot.netcode import (
    BTCategory,
    PlexAPIClient,
    QBittorrentClient,
    TorrentInfoResponse,
)

from bot.tools import (
    check_for_album,
    check_for_movie,
    search_for_torrent,
    add_torrent,
    TorrentContext,
)
from bot.config import Config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
server_guild = discord.Object(id=Config.DISCORD_GUILD_ID)


@bot.event
async def on_ready():
    logger.info(f"{bot.user} is running in {Config.ENVIRONMENT} mode")
    logger.info(f"Bot is in {len(bot.guilds)} guilds")

    # Sync slash commands with Discord
    try:
        num_commands = await bot.tree.sync()
        logger.info(f"{len(num_commands)} Slash commands synced globally")
        num_commands = await bot.tree.sync(guild=server_guild)
        logger.info(f"{len(num_commands)} Slash commands synced in server guild")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")


@bot.tree.command(name="ping", description="Check if the bot is responsive")
async def ping(interaction: discord.Interaction):
    logger.info("Received ping command")
    await interaction.response.send_message(
        f"Pong! Running in {Config.ENVIRONMENT} mode"
    )


@bot.tree.command(name="query", description="Search for media")
@discord.app_commands.describe(query="The query to send to chatgpt")
async def query(interaction: discord.Interaction, query: str):
    load_dotenv()
    llm = ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
    )  # type: ignore

    agent = create_agent(
        llm,
        tools=[search_for_torrent, add_torrent, check_for_album, check_for_movie],
        context_schema=TorrentContext,
    )
    await interaction.response.defer()
    torrent_context = TorrentContext(
        search_results={}, internal_torrents={}, torrent_types=set()
    )
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant that can search for torrents using qBittorrent.
                    You should interpret whether the user wants a movie or an album, and use the appropriate tools.
                    Based on the user's query, use the tool to find relevant torrents.
                    Select the most appropriate torrents, prefer higher quality, and only choose a vinyl rip if the user specifically requests it.
                    Provide the user with a concise summary of the top results.
                    Then use the add_torrent tool to add the selected torrents.
                    Some rules:
                        - Do not ask follow up questions. Assume the user wants all torrents available.
                        - If two torrents are similar enough that they may be the same album but one is a special release, only get the special release.
                        - Do not ever download the same album in two formats.
                        """,
                },
                {"role": "user", "content": query},
            ]
        },
        context=torrent_context,
    )
    logger.info(f"Agent response: {response}")
    await interaction.followup.send(response["messages"][-1].content)
    # Get torrent info for all added torrents, send a message when all of them are ready
    torrent_sync_targets: dict[str, BTCategory] = {}
    torrent_info: list[TorrentInfoResponse] = []
    while True:
        async with QBittorrentClient() as qclient:
            for torrent in torrent_context.internal_torrents:
                torrent_info_promises = []
                for (
                    content_path,
                    memory_code,
                ) in torrent_context.internal_torrents.items():
                    if memory_code is None:
                        continue
                    torrent_info_promises.append(qclient.get_torrent_info(memory_code))
                torrent_info = await asyncio.gather(*torrent_info_promises)
                logger.info(f"Torrent info: {torrent_info}")
            if all([info.progress == 1.0 for info in torrent_info]):
                break
            for info in torrent_info:
                torrent_sync_targets[info.content_path] = BTCategory[info.category]
            await asyncio.sleep(1)

    # Trigger a plex sync
    async with PlexAPIClient() as plex_client:
        for content_path, category in torrent_sync_targets.items():
            await plex_client.scan_media(content_path, category)

    await interaction.followup.send(
        f"""
    The following files have been added to the server:\n-
    {'\n- '.join(torrent_context.internal_torrents.keys())}
    """
    )


if __name__ == "__main__":
    Config.validate()
    bot.run(Config.DISCORD_TOKEN)
