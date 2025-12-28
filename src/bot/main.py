import logging
from argparse import ArgumentParser
import os

import asyncio
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import discord
from discord.ext import commands

from bot.tools import check_for_album, search_for_albums, add_torrent, KnownTorrents
from bot.config import Config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="sprintboy.log",
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
        await bot.tree.sync(guild=server_guild)
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")


@bot.tree.command(name="ping", description="Check if the bot is responsive")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Pong! Running in {Config.ENVIRONMENT} mode"
    )


@bot.tree.command(name="music", description="Search for albums")
@discord.app_commands.describe(query="The query to send to chatgpt")
async def music_query(interaction: discord.Interaction, query: str):
    load_dotenv()
    llm = ChatOllama(
        model="gpt-oss:120b",
        base_url=Config.OLLAMA_API_URL,
        temperature=0,
        num_ctx=32000,
    )
    # llm = ChatAnthropic(
    # model_name="claude-sonnet-4-5-20250929",
    # )

    agent = create_agent(
        llm,
        tools=[search_for_albums, add_torrent, check_for_album],
        context_schema=KnownTorrents,
    )
    await interaction.response.defer()
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can search for torrents using qBittorrent. Based on the user's query, use the tool to find relevant torrents. Select the most appropriate torrents, prefer higher quality, and only choose a vinyl rip if the user specifically requests it. Provide the user with a concise summary of the top results. Then use the add_torrent tool to add the selected torrents. Some rules: \n - Do not ask follow up questions. Assume the user wants all torrents available. \n - If two torrents are similar enough that they may be the same album but one is a special release, only get the special release. \n - Do not ever download the same album in two formats.",
                },
                {"role": "user", "content": query},
            ]
        },
        context=KnownTorrents(torrents={}),
    )
    logger.info(f"Agent response: {response}")
    await interaction.followup.send(response["messages"][-1].content)


if __name__ == "__main__":
    Config.validate()
    bot.run(Config.DISCORD_TOKEN)
