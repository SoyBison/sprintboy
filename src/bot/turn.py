import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable

from bot import workflows
from bot.agent import AgentResult, run_agent
from bot.decide import current_run_id
from bot.llm import ChatModel, Message, build_chat_model, from_dict, system, user
from bot.output import add_nudge
from bot.routing import MIN_DOMAIN_CONFIDENCE, Route, route
from bot.toolkit import Tool
from bot.tools import (
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
    if route is None or not route.trusted or route.domain == "tracker":
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


_model: ChatModel | None = None


def get_model() -> ChatModel:
    """Build the chat model once and reuse it across messages."""
    global _model
    if _model is None:
        _model = build_chat_model()
        logger.info(f"Using {type(_model).__name__} model {_model.name}")
    return _model


def _earlier(history: list[dict]) -> list[dict]:
    """The conversation before the latest user message."""
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "user":
            return history[:index]
    return []


def latest_user_text(history: list[dict]) -> str:
    for entry in reversed(history):
        if entry.get("role") == "user":
            return entry.get("content", "")
    return ""


@dataclass
class Turn:
    tools: list[Tool]
    messages: list[Message]
    route: Route | None
    run_id: str
    text: str = ""
    earlier: list[dict] = field(default_factory=list)


async def prepare(history: list[dict]) -> Turn:
    """Route the latest message, then pick the tools, messages and run id."""
    run_id = uuid.uuid4().hex[:12]
    current_run_id.set(run_id)
    text = latest_user_text(history)
    try:
        decided = await route(text, run_id=run_id, earlier=_earlier(history))
    except Exception:
        logger.warning("Routing failed; using all tools", exc_info=True)
        decided = None
    note = decided.note() if decided is not None else ""
    content = SYSTEM_PROMPT + ("\n\n" + note if note else "")
    tools = select_tools(decided)
    logger.info(f"Route {run_id}: {decided} tools={[t.name for t in tools]}")
    return Turn(
        tools=tools,
        messages=[system(content), *[from_dict(h) for h in history]],
        route=decided,
        run_id=run_id,
        text=text,
        earlier=_earlier(history),
    )


async def run(
    turn: Turn, context, on_event: Callable | None = None, use_workflows: bool = True
) -> AgentResult:
    """Run the agent, with one follow-up if a download turn added nothing.

    Requests with a known shape go to a workflow first; it returns None when it
    cannot handle the message, and the agent takes over. `use_workflows=False`
    goes straight to the agent.
    """
    decided = turn.route
    # Kind does not matter for the account workflow: it asks its own question.
    if (
        use_workflows
        and decided
        and decided.domain == "tracker"
        and decided.domain_p >= MIN_DOMAIN_CONFIDENCE
    ):
        started = time.perf_counter()
        try:
            wf = await workflows.account(turn.text, run_id=turn.run_id)
        except Exception:
            logger.exception(f"Run {turn.run_id}: account workflow failed")
            wf = None
        if wf is not None:
            if on_event:
                emitted = on_event("reply", {"content": wf.reply})
                if inspect.isawaitable(emitted):
                    await emitted
            logger.info(
                f"Run {turn.run_id}: handled by the account workflow "
                f"in {time.perf_counter() - started:.1f}s"
            )
            return wf.to_agent_result(turn.messages)
    if use_workflows and decided and decided.trusted and decided.domain == "music":
        if decided.kind == "specific":
            name = "specific"

            def start():
                return workflows.specific(
                    turn.text, context, earlier=turn.earlier, run_id=turn.run_id
                )

        elif decided.kind == "discography":
            name = "discography"

            def start():
                return workflows.discography(turn.text, context, run_id=turn.run_id)

        elif decided.kind == "open_ended":
            name = "recommend"

            def start():
                return workflows.recommend(
                    turn.text,
                    context,
                    count=decided.count or 3,
                    earlier=turn.earlier,
                    run_id=turn.run_id,
                )

        else:
            start = None
        if start is not None:
            started = time.perf_counter()
            try:
                wf = await start()
            except Exception:
                logger.exception(f"Run {turn.run_id}: {name} workflow failed")
                wf = None
            if wf is not None:
                if on_event:
                    emitted = on_event("reply", {"content": wf.reply})
                    if inspect.isawaitable(emitted):
                        await emitted
                logger.info(
                    f"Run {turn.run_id}: handled by the {name} workflow "
                    f"in {time.perf_counter() - started:.1f}s"
                )
                return wf.to_agent_result(turn.messages)
    known = set(context.internal_torrents)
    model = get_model()
    result = await run_agent(
        model, turn.messages, turn.tools, context, run_id=turn.run_id, on_event=on_event
    )
    new_torrents = [n for n in context.internal_torrents if n not in known]
    nudge = add_nudge(result.messages, new_torrents, turn.route)
    # A run that used every step already tried hard; another 12 is not a nudge.
    if not nudge or result.stopped == "max_steps":
        return result
    logger.info(f"Run {turn.run_id}: nudging once")
    second = await run_agent(
        model,
        [*result.messages, user(nudge)],
        turn.tools,
        context,
        run_id=turn.run_id,
        on_event=on_event,
    )
    return replace(second, steps=[*result.steps, *second.steps], nudged=True)
