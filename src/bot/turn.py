import logging
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent

from bot.config import Config
from bot.decide import current_run_id
from bot.routing import Route, route
from bot.tools import (
    TorrentContext,
    add_torrent,
    download_albums,
    check_albums,
    check_for_album,
    check_for_movie,
    lastfm_artist_albums,
    lastfm_artist_info,
    lastfm_browse_tag,
    lastfm_resolve,
    lastfm_similar_artists,
    search_for_torrent,
)

logger = logging.getLogger(__name__)

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
    3. Call download_albums ONCE with every album you picked (artist and title, Last.fm
       spelling). It checks the library, finds the best version and adds it, so you do not
       need search_for_torrent or add_torrent for music unless it reports NOT FOUND and you
       want to look for a differently named release.
    4. For an open-ended request, if some albums come back OWNED, NOT FOUND or NOT ADDED,
       call download_albums again with replacement albums until you reach the count.

Rules:
    - Do not ask follow up questions. Make a reasonable choice and act on it.
    - Always use the batch form of tools (several albums per call) instead of one call per
      album. It is much faster.
    - Searching is not downloading. Never say you added, downloaded or grabbed something
      unless add_torrent or download_albums answered "Added" for it. If it answered "NOT ADDED", say plainly
      that it is not downloading and why.
    - download_albums and search_for_torrent already pick the best version (SACD and
      perfect CD rips first, never vinyl). Only pass media ("CD", "SACD", "WEB" or "Vinyl")
      when the user asks for that source. For movies and TV, prefer higher quality.
    - If two torrents are the same album but one is a special or deluxe release, get only the
      special release. Never download the same album twice in two formats.
    - Prefer full studio albums over EPs, singles, live records and compilations unless asked.
    - For open-ended requests, spread picks across different artists (at most two albums per
      artist) unless they asked for more by one artist.
    - If a Last.fm tool errors, say so and carry on with the torrent search, but do not
      substitute guessed album names for ones you could not verify.
    - If this prompt ends with a "Routing:" line, trust it over your own reading of the request.

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
    download_albums,
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


AGENT_TOOLS = [
    download_albums,
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

MUSIC_READ = [
    lastfm_artist_info,
    lastfm_similar_artists,
    lastfm_artist_albums,
    lastfm_browse_tag,
    lastfm_resolve,
    check_albums,
    check_for_album,
]
MOVIE_READ = [check_for_movie]
DOWNLOAD = [search_for_torrent, add_torrent]
MUSIC_DOWNLOAD = [download_albums, search_for_torrent, add_torrent]


def select_tools(route: Route | None) -> list:
    """Only the tools the routed request needs; everything if we are unsure."""
    if route is None or not route.trusted:
        return AGENT_TOOLS
    if route.kind == "question":
        if route.domain == "music":
            return MUSIC_READ
        if route.domain in ("movie", "tv"):
            return MOVIE_READ
        return MUSIC_READ + MOVIE_READ
    if route.domain == "music":
        return MUSIC_READ + MUSIC_DOWNLOAD
    if route.domain in ("movie", "tv"):
        return MOVIE_READ + DOWNLOAD
    return AGENT_TOOLS


_llm = None
_agents: dict[tuple[str, ...], object] = {}


def get_agent(tools):
    """Build agents once per tool set and reuse them across messages."""
    global _llm
    key = tuple(t.name for t in tools)
    agent = _agents.get(key)
    if agent is None:
        load_dotenv()
        if _llm is None:
            _llm = build_llm()
        agent = create_agent(_llm, tools=tools, context_schema=TorrentContext)
        _agents[key] = agent
    return agent


def latest_user_text(history: list[dict]) -> str:
    for entry in reversed(history):
        if entry.get("role") == "user":
            return entry.get("content", "")
    return ""


async def prepare(history: list[dict]):
    """Route the latest message, then pick the agent, messages and run id."""
    run_id = uuid.uuid4().hex[:12]
    current_run_id.set(run_id)
    try:
        decided = await route(latest_user_text(history), run_id=run_id)
    except Exception:
        logger.warning("Routing failed; using all tools", exc_info=True)
        decided = None
    note = decided.note() if decided is not None else ""
    content = SYSTEM_PROMPT + ("\n\n" + note if note else "")
    tools = select_tools(decided)
    logger.info(f"Route {run_id}: {decided} tools={[t.name for t in tools]}")
    return (
        get_agent(tools),
        [{"role": "system", "content": content}, *history],
        decided,
        run_id,
    )
