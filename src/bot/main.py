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
COMMAND_PREFIX = "!"
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)
server_guild = discord.Object(id=Config.DISCORD_GUILD_ID)

DISCORD_MESSAGE_LIMIT = 2000
# How far back up a reply chain we rebuild a conversation.
MAX_HISTORY_DEPTH = 20
# How many conversations we keep torrent context for.
MAX_TRACKED_CONVERSATIONS = 50

SYSTEM_PROMPT = """
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
"""

# Keyed by the id of the message that started a conversation, so that a reply
# chain keeps the search results and downloads it has accumulated so far.
conversation_contexts: dict[int, TorrentContext] = {}

_agent = None


def get_agent():
    """Build the agent once and reuse it across messages."""
    global _agent
    if _agent is None:
        load_dotenv()
        llm = ChatAnthropic(
            model_name="claude-sonnet-4-5-20250929",
        )  # type: ignore
        _agent = create_agent(
            llm,
            tools=[search_for_torrent, add_torrent, check_for_album, check_for_movie],
            context_schema=TorrentContext,
        )
    return _agent


def get_conversation_context(root_id: int) -> TorrentContext:
    context = conversation_contexts.get(root_id)
    if context is None:
        context = TorrentContext(
            search_results={}, internal_torrents={}, torrent_types=set()
        )
        conversation_contexts[root_id] = context
        while len(conversation_contexts) > MAX_TRACKED_CONVERSATIONS:
            conversation_contexts.pop(next(iter(conversation_contexts)))
    return context


def message_text(content) -> str:
    """Flatten an LLM message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(part for part in parts if part)
    return str(content)


async def build_conversation(
    message: discord.Message,
) -> tuple[list[dict[str, str]], int]:
    """Walk the reply chain upwards to rebuild the conversation.

    A message with no reply history starts a new line of questioning, so the
    chain is just the message itself.
    """
    chain: list[discord.Message] = []
    current: discord.Message | None = message
    while current is not None and len(chain) < MAX_HISTORY_DEPTH:
        chain.append(current)
        reference = current.reference
        if reference is None or reference.message_id is None:
            break
        resolved = reference.resolved
        if isinstance(resolved, discord.Message):
            current = resolved
            continue
        try:
            current = await current.channel.fetch_message(reference.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            break
    chain.reverse()

    history = [
        {
            "role": "assistant" if m.author == bot.user else "user",
            "content": m.clean_content,
        }
        for m in chain
        if m.clean_content.strip()
    ]
    return history, chain[0].id


async def reply_in_chunks(message: discord.Message, text: str) -> discord.Message:
    """Reply to a message, splitting over Discord's length limit if needed.

    Each chunk replies to the previous one so the conversation stays a single
    chain, and returns the last message sent.
    """
    text = text.strip() or "Done."
    target = message
    for start in range(0, len(text), DISCORD_MESSAGE_LIMIT):
        target = await target.reply(
            text[start : start + DISCORD_MESSAGE_LIMIT], mention_author=False
        )
    return target


async def wait_for_downloads(
    message: discord.Message, torrent_context: TorrentContext, new_torrents: list[str]
):
    """Poll qBittorrent until the newly added torrents finish, then sync Plex."""
    torrent_sync_targets: dict[str, BTCategory] = {}
    torrent_info: list[TorrentInfoResponse] = []
    while True:
        async with QBittorrentClient() as qclient:
            torrent_info_promises = []
            for name in new_torrents:
                memory_code = torrent_context.internal_torrents.get(name)
                if memory_code is None:
                    continue
                torrent_info_promises.append(qclient.get_torrent_info(memory_code))
            torrent_info = await asyncio.gather(*torrent_info_promises)
            logger.info(f"Torrent info: {torrent_info}")
            # Record sync targets before checking for completion, otherwise a
            # torrent that is already done on the first poll never gets scanned.
            for info in torrent_info:
                # Look up by value, not name: qBittorrent stores the category
                # capitalized, so "TV" comes back as "Tv" and name lookup fails.
                torrent_sync_targets[info.content_path] = BTCategory(
                    info.category.lower()
                )
            if all([info.progress == 1.0 for info in torrent_info]):
                break
            await asyncio.sleep(1)

    # Trigger a plex sync
    async with PlexAPIClient() as plex_client:
        for content_path, category in torrent_sync_targets.items():
            await plex_client.scan_media(content_path, category)

    await reply_in_chunks(
        message,
        f"""
    The following files have been added to the server:\n
    {"\n - ".join(new_torrents)}
    """,
    )


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


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not message.clean_content.strip():
        return
    if message.content.startswith(COMMAND_PREFIX):
        await bot.process_commands(message)
        return
    if (
        Config.DISCORD_CHANNEL_IDS
        and str(message.channel.id) not in Config.DISCORD_CHANNEL_IDS
    ):
        return

    history, root_id = await build_conversation(message)
    torrent_context = get_conversation_context(root_id)
    known_torrents = set(torrent_context.internal_torrents)
    logger.info(f"Handling message {message.id} in conversation {root_id}")

    async with message.channel.typing():
        try:
            response = await get_agent().ainvoke(
                {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history]},
                context=torrent_context,
            )
        except Exception as e:
            logger.exception("Agent invocation failed")
            await reply_in_chunks(message, f"Something went wrong: {e}")
            return

    logger.info(f"Agent response: {response}")
    last_message = await reply_in_chunks(
        message, message_text(response["messages"][-1].content)
    )

    new_torrents = [
        name for name in torrent_context.internal_torrents if name not in known_torrents
    ]
    if new_torrents:
        await wait_for_downloads(last_message, torrent_context, new_torrents)


if __name__ == "__main__":
    Config.validate()
    bot.run(Config.DISCORD_TOKEN)
