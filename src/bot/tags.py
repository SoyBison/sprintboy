"""
Last.fm's tag vocabulary, cached on disk.

A request like "throw some new shoegaze at me" or "something chill to work to"
is asked against a shortlist of real Last.fm tags, so the decision model picks
a tag that `tag.getTopAlbums` actually knows. The list changes slowly, so it is
fetched once a week and kept next to the decision log.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from thefuzz import fuzz

from bot.config import Config
from bot.netcode import LastFMClient

logger = logging.getLogger(__name__)

CACHE_MAX_AGE = timedelta(days=7)
KEEP_TAGS = 200
FETCH_TAGS = 500
MIN_FUZZY_LENGTH = 4
FUZZY_THRESHOLD = 85

FALLBACK_TAGS = [
    "rock", "indie", "jazz", "electronic", "ambient", "chill", "chillout", "shoegaze",
    "dream pop", "post-rock", "hip-hop", "soul", "funk", "psychedelic", "folk", "metal",
    "punk", "emo", "city pop", "jazz fusion", "lo-fi", "downtempo", "house", "techno",
    "classical", "experimental", "singer-songwriter", "r&b", "blues", "country",
    "reggae", "krautrock", "synthpop", "new wave", "post-punk", "trip-hop",
    "bossa nova", "afrobeat", "math rock", "noise rock",
]

# Tags about the listener or the artist's passport rather than the sound.
# Decades stay: "from the 60s" is a real request. Compound genres such as
# "japanese city pop" are not in this set, so they stay too.
NOISE_TAGS = {
    "seen live", "favorites", "favourites", "favorite", "favourite", "favorite albums",
    "female vocalists", "male vocalists", "female vocalist", "male vocalist",
    "british", "american", "uk", "usa", "german", "canadian", "australian",
    "swedish", "japanese", "french", "italian", "irish", "scottish", "english",
    "norwegian", "finnish", "dutch", "spanish", "brazilian", "russian", "polish",
    "albums i own", "beautiful", "awesome", "love", "spotify", "amazing", "good",
    "all", "best", "cool", "lovely",
}


def _cache_path() -> Path:
    return Path(Config.DECISION_LOG_PATH).parent / "lastfm_tags.json"


def _clean(tags: list[tuple[str, int]]) -> list[str]:
    seen: dict[str, int] = {}
    for name, reach in tags:
        tag = " ".join(name.casefold().split())
        if tag and tag not in NOISE_TAGS and tag not in seen:
            seen[tag] = reach
    ranked = sorted(seen, key=lambda t: -seen[t])
    return ranked[:KEEP_TAGS]


def _read_cache(path: Path) -> tuple[datetime, list[str]] | None:
    try:
        data = json.loads(path.read_text())
        fetched = datetime.fromisoformat(data["fetched"])
        tags = [str(t) for t in data["tags"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (fetched, tags) if tags else None


async def tag_vocabulary() -> list[str]:
    """Popular genre and mood tags, most popular first."""
    path = _cache_path()
    cached = _read_cache(path)
    now = datetime.now(timezone.utc)
    if cached and now - cached[0] < CACHE_MAX_AGE:
        return cached[1]
    try:
        async with LastFMClient() as lastfm:
            tags = _clean(await lastfm.get_top_tags(FETCH_TAGS))
        if not tags:
            raise ValueError("Last.fm returned no tags")
    except Exception as e:
        logger.warning(f"Could not refresh the Last.fm tag list: {e}")
        return cached[1] if cached else list(FALLBACK_TAGS)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched": now.isoformat(), "tags": tags}))
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")
    return tags


def vocabulary_matches(text: str, vocab: list[str], limit: int = 40) -> list[str]:
    """The tags worth offering for this message: named in it, close to it, then popular ones."""
    lowered = text.casefold()
    chosen: dict[str, None] = {}
    for tag in vocab:
        if re.search(rf"(?<!\w){re.escape(tag.casefold())}(?!\w)", lowered):
            chosen.setdefault(tag)

    words = [w for w in re.findall(r"[\w&'-]+", lowered)]
    spans = {
        " ".join(words[i : i + n])
        for n in (1, 2, 3)
        for i in range(len(words) - n + 1)
        if any(len(w) >= 3 for w in words[i : i + n])
    }
    scored = []
    for tag in vocab:
        if tag in chosen or len(tag) < MIN_FUZZY_LENGTH:
            continue
        best = max((fuzz.partial_ratio(tag.casefold(), span) for span in spans), default=0)
        if best >= FUZZY_THRESHOLD:
            scored.append((best, tag))
    for _, tag in sorted(scored, key=lambda s: -s[0]):
        chosen.setdefault(tag)

    for tag in vocab:
        chosen.setdefault(tag)
    return list(chosen)[:limit]
