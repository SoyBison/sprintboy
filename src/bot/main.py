import asyncio
import logging

import discord
from discord.ext import commands
from bot.netcode import (
    BTCategory,
    PlexAPIClient,
    QBittorrentClient,
    TorrentInfoResponse,
)

from bot.tools import TorrentContext
from bot import aotm, turn
from bot.turn import SYSTEM_PROMPT, AGENT_TOOLS, build_llm  # noqa: F401
from bot.output import (
    best_reply,
    describe_failure,
    name_list,
    add_nudge,
    split_for_discord,
    summarise_run,
    unfulfilled_note,
)
from bot.config import Config, git_sha, setup_logging

# Honour LOG_LEVEL: this was pinned to DEBUG, which made the deployed logs
# unreadable and rolled the container's 50m of history in minutes.
setup_logging()
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

# Keyed by the id of the message that started a conversation, so that a reply
# chain keeps the search results and downloads it has accumulated so far.
conversation_contexts: dict[int, TorrentContext] = {}

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
    target = message
    for chunk in split_for_discord(text.strip(), DISCORD_MESSAGE_LIMIT) or ["Done."]:
        target = await target.reply(chunk, mention_author=False)
    return target


async def wait_for_downloads(
    message: discord.Message, torrent_context: TorrentContext, new_torrents: list[str]
):
    """Poll qBittorrent until the newly added torrents finish, then sync Plex."""
    torrent_sync_targets: dict[str, BTCategory] = {}
    torrent_info: list[TorrentInfoResponse] = []
    waiting_on = [
        name for name in new_torrents if torrent_context.internal_torrents.get(name)
    ]
    if not waiting_on:
        # `all([])` is True, so without this an empty poll would fall straight
        # through to announcing a finished download that never started.
        logger.warning(f"Nothing to wait for out of {new_torrents}")
        return
    while True:
        async with QBittorrentClient() as qclient:
            torrent_info_promises = []
            for name in waiting_on:
                memory_code = torrent_context.internal_torrents[name]
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
        f"Finished downloading, and Plex has been told to scan:\n"
        f"{name_list(waiting_on)}",
    )


async def _aotm_added(message: discord.Message, name: str, memory_code: str):
    """Wait for an Album of the Month grab to finish, then have Plex scan it."""
    context = TorrentContext(
        search_results={},
        internal_torrents={name: memory_code},
        torrent_types={BTCategory.Music},
    )
    try:
        await wait_for_downloads(message, context, [name])
    except Exception as e:
        logger.exception("Waiting for the AoTM download failed")
        await reply_in_chunks(
            message,
            f"The download was added but I lost track of it, so Plex has "
            f"not been told to scan. {describe_failure(e)}",
        )


aotm_loop = aotm.setup(bot, _aotm_added)


@bot.event
async def on_ready():
    logger.info(
        f"{bot.user} is running in {Config.ENVIRONMENT} mode on commit {git_sha()}"
    )
    logger.info(f"Bot is in {len(bot.guilds)} guilds")

    # Sync slash commands with Discord
    try:
        num_commands = await bot.tree.sync()
        logger.info(f"{len(num_commands)} Slash commands synced globally")
        num_commands = await bot.tree.sync(guild=server_guild)
        logger.info(f"{len(num_commands)} Slash commands synced in server guild")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

    if not aotm_loop.is_running():
        aotm_loop.start()


@bot.tree.command(name="ping", description="Check if the bot is responsive")
async def ping(interaction: discord.Interaction):
    logger.info("Received ping command")
    await interaction.response.send_message(
        f"Pong! Running in {Config.ENVIRONMENT} mode on commit `{git_sha()}`"
    )


def _new(context: TorrentContext, known: set[str]) -> list[str]:
    """Torrents added since `known` was snapshotted."""
    return [name for name in context.internal_torrents if name not in known]


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
    run_id = None

    async with message.channel.typing():
        try:
            agent, messages, route, run_id = await turn.prepare(history)
            logger.info(
                f"Handling message {message.id} in conversation {root_id} run {run_id}"
            )
            response = await agent.ainvoke(
                {"messages": messages},
                context=torrent_context,
            )
            nudge = add_nudge(
                response["messages"], _new(torrent_context, known_torrents), route
            )
            if nudge:
                logger.info(f"Run {run_id}: download turn added nothing; nudging once")
                response = await agent.ainvoke(
                    {
                        "messages": [
                            *response["messages"],
                            {"role": "user", "content": nudge},
                        ]
                    },
                    context=torrent_context,
                )
        except Exception as e:
            logger.exception("Agent invocation failed")
            await reply_in_chunks(message, describe_failure(e))
            return

    logger.info(f"Run {run_id}: {summarise_run(response['messages'])}")
    new_torrents = _new(torrent_context, known_torrents)
    reply = best_reply(response["messages"], new_torrents)
    note = unfulfilled_note(response["messages"], new_torrents)
    if note:
        logger.warning(f"Correcting an unfulfilled reply: {reply[:200]!r}")
        reply = f"{reply}\n\n**{note}**"
    last_message = await reply_in_chunks(message, reply)

    if new_torrents:
        try:
            await wait_for_downloads(last_message, torrent_context, new_torrents)
        except Exception as e:
            # Raising here only reaches discord.py's logger, which leaves the
            # last word in the chat being that the download had started.
            logger.exception("Waiting for the downloads failed")
            await reply_in_chunks(
                last_message,
                f"The downloads were added but I lost track of them, so Plex has "
                f"not been told to scan. {describe_failure(e)}",
            )


if __name__ == "__main__":
    Config.validate()
    bot.run(Config.DISCORD_TOKEN)
