import asyncio
import logging

from langchain.agents import create_agent
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
    lastfm_artist_albums,
    lastfm_artist_info,
    lastfm_browse_tag,
    lastfm_resolve,
    lastfm_similar_artists,
    search_for_torrent,
    add_torrent,
    TorrentContext,
)
from bot.output import (
    best_reply,
    describe_failure,
    name_list,
    split_for_discord,
    summarise_run,
)
from bot.config import Config, setup_logging

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

SYSTEM_PROMPT = """
You are a helpful assistant that adds music, movies and TV to the user's Plex server by
searching for torrents in qBittorrent.
First work out whether the user wants music, a movie, or a TV show, and use the tools for that kind of media.

For music, follow this order:
    1. Establish what actually exists using the Last.fm tools. Do not rely on your own
       recollection of an artist's discography, album titles or which artists sound alike:
       your training data is stale and you will invent releases that were never made.
       - lastfm_artist_albums for an artist's real releases and their real spellings
       - lastfm_similar_artists and lastfm_browse_tag for recommendations and styles
       - lastfm_artist_info to learn what an artist sounds like and which tags to follow
       - lastfm_resolve to turn a vague or misspelled name into the real release name
    2. Call check_for_album for every album you are considering. It canonicalises the
       names through Last.fm and fuzzy matches the whole library, so trust its answer:
       if it reports a COLLISION, drop that album and say so. If it reports a possible
       match, decide for yourself whether it is the same release.
    3. Search for the remaining albums with search_for_torrent, using the Last.fm
       spelling of the artist and album.
    4. Add the ones you picked with add_torrent, then summarise concisely what you added
       and what you skipped because the user already had it.

Some rules:
    - Do not ask follow up questions. Assume the user wants all torrents available.
    - Prefer higher quality, and only choose a vinyl rip if the user specifically requests it.
    - If two torrents are similar enough that they may be the same album but one is a special release, only get the special release.
    - Do not ever download the same album in two formats.
    - If a Last.fm tool errors, say so and carry on with the torrent search rather than
      giving up, but do not substitute guessed album names for the ones you could not verify.

Your reply is posted straight into a Discord chat, so:
    - Always finish with a reply. Never stop after a tool call without saying what happened.
    - Write it to the person who asked, as "you", and never refer to them as "the user".
    - Say what you added and what you skipped, naming the albums. If you added nothing, say
      why in one line.
    - No preamble, no restating the request, no notes about your own process, tool names or
      reasoning, and no lists of steps you are about to take.
    - Keep it short: a sentence or two, plus a list of album names if there is one. Leave out
      any list that would be empty.
"""

# Keyed by the id of the message that started a conversation, so that a reply
# chain keeps the search results and downloads it has accumulated so far.
conversation_contexts: dict[int, TorrentContext] = {}

_agent = None


def build_llm():
    """Build the chat model for the configured backend.

    Imported lazily so that running on one backend does not require the other's
    package to be installed or its credentials to be present.
    """
    Config.validate_llm()
    if Config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        logger.info(
            f"Using ollama model {Config.OLLAMA_MODEL} at {Config.OLLAMA_API_URL}"
        )
        return ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_API_URL,
            num_ctx=Config.OLLAMA_NUM_CTX,
            # The agent picks between concrete torrents and album titles, so a
            # creative model just invents releases that were not in the results.
            temperature=0,
            # Fail at startup with a clear message rather than on the first
            # message with a 404 from a model that was never pulled.
            validate_model_on_init=Config.OLLAMA_VALIDATE_MODEL,
        )

    from langchain_anthropic import ChatAnthropic

    logger.info(f"Using anthropic model {Config.ANTHROPIC_MODEL}")
    return ChatAnthropic(model_name=Config.ANTHROPIC_MODEL)  # type: ignore


AGENT_TOOLS = [
    search_for_torrent,
    add_torrent,
    check_for_album,
    check_for_movie,
    lastfm_artist_info,
    lastfm_similar_artists,
    lastfm_artist_albums,
    lastfm_browse_tag,
    lastfm_resolve,
]


def get_agent():
    """Build the agent once and reuse it across messages."""
    global _agent
    if _agent is None:
        load_dotenv()
        _agent = create_agent(
            build_llm(),
            tools=AGENT_TOOLS,
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
        f"Finished downloading, and Plex has been told to scan:\n"
        f"{name_list(new_torrents)}",
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
            await reply_in_chunks(message, describe_failure(e))
            return

    logger.info(summarise_run(response["messages"]))
    new_torrents = [
        name for name in torrent_context.internal_torrents if name not in known_torrents
    ]
    last_message = await reply_in_chunks(
        message, best_reply(response["messages"], new_torrents)
    )

    if new_torrents:
        await wait_for_downloads(last_message, torrent_context, new_torrents)


if __name__ == "__main__":
    Config.validate()
    bot.run(Config.DISCORD_TOKEN)
