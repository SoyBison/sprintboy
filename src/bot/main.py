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
    check_albums,
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
    ADD_NUDGE,
    describe_failure,
    name_list,
    needs_add_nudge,
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

SYSTEM_PROMPT = """
You add music, movies and TV to the user's Plex server by finding torrents through qBittorrent.
First work out whether they want music, a movie, or a TV show, and use the tools for that kind of media.

Before doing anything, decide which kind of request this is:
    A. SPECIFIC: named albums, "the new X album", "X's discography", "fill in my X collection".
       The user wants exactly those releases. Never download anything they already own; if
       they own all of it, adding nothing is the correct result, so say so.
    B. OPEN-ENDED: "5 albums like X", "something new", "some shoegaze", "surprise me",
       "more by X". The user wants NEW music. Albums they already own do not count towards
       what they asked for: skip them silently and pick something else. Keep going until you
       have added the number they asked for (3 if they gave no number). Only stop short if you
       have genuinely run out of candidates, and say how many you managed.

For music:
    1. Find real candidates with Last.fm. Never rely on your own memory of discographies,
       album titles or similar artists: it is stale and you will invent releases.
       - lastfm_artist_albums for real releases (pass several artists in one call)
       - lastfm_similar_artists and lastfm_browse_tag for recommendations and styles
       - lastfm_artist_info to learn what an artist sounds like and which tags to follow
       - lastfm_resolve to turn a vague or misspelled name into the real release name
       These results are marked [OWNED], [maybe owned as ...], [in library: N albums] or
       [new to you]. Trust the marks: drop [OWNED] albums, and for open-ended requests prefer
       artists marked [new to you]. For open-ended requests, gather about twice as many
       candidates as you need so that owned or unavailable ones can be replaced.
    2. For a SPECIFIC request, check every album you are about to search for in ONE
       check_albums call (it catches aliases and editions the marks can miss). For an
       open-ended request the marks are enough; only check albums that came from elsewhere.
       Never call check_albums or check_for_album once per album.
       A COLLISION means they own it: drop it. A possible match means decide yourself whether it is the same release.
    3. Search for ALL remaining candidates in ONE search_for_torrent call, one query per
       album, as "Artist Album" using the Last.fm spelling.
    4. Add every torrent you picked in ONE add_torrent call, using the names exactly as the
       search returned them.
    5. For an open-ended request, if some candidates had no torrent or failed to add, repeat
       steps 3-4 with the next candidates until you reach the count. Do not stop after one.

Rules:
    - Do not ask follow up questions. Make a reasonable choice and act on it.
    - Always use the batch form of tools (several albums per call) instead of one call per
      album. It is much faster.
    - Searching is not downloading. Never say you added, downloaded or grabbed something
      unless add_torrent answered "Added" for it. If it answered "NOT ADDED", say plainly
      that it is not downloading and why.
    - Prefer higher quality (24bit over 16bit, lossless over lossy), and only choose a vinyl
      rip if the user specifically asks for one.
    - If two torrents are the same album but one is a special or deluxe release, get only the
      special release. Never download the same album twice in two formats.
    - Prefer full studio albums over EPs, singles, live records and compilations unless asked.
    - For open-ended requests, spread picks across different artists (at most two albums per
      artist) unless they asked for more by one artist.
    - If a Last.fm tool errors, say so and carry on with the torrent search, but do not
      substitute guessed album names for ones you could not verify.

Your reply is posted straight into a Discord chat, so:
    - Always finish with a reply. Never stop after a tool call without saying what happened.
    - Write it to the person who asked, as "you", and never refer to them as "the user".
    - List what you added. For specific requests, also name what you skipped because they
      already had it. For open-ended requests, do not list owned albums you passed over.
      If you added nothing, say why in one line.
    - No preamble, no restating the request, no notes about your own process, tool names or
      reasoning, and no lists of steps you are about to take.
    - Keep it short: a sentence or two, plus a list of album names if there is one.
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
    check_albums,
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
    logger.info(f"Handling message {message.id} in conversation {root_id}")

    async with message.channel.typing():
        try:
            response = await get_agent().ainvoke(
                {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history]},
                context=torrent_context,
            )
            if needs_add_nudge(response["messages"], _new(torrent_context, known_torrents)):
                logger.info("Turn searched but added nothing; nudging once")
                response = await get_agent().ainvoke(
                    {
                        "messages": [
                            *response["messages"],
                            {"role": "user", "content": ADD_NUDGE},
                        ]
                    },
                    context=torrent_context,
                )
        except Exception as e:
            logger.exception("Agent invocation failed")
            await reply_in_chunks(message, describe_failure(e))
            return

    logger.info(summarise_run(response["messages"]))
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
