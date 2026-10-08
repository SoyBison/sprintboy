"""Parse and rank music tracker result names so the best version is chosen in code."""

import re
from dataclasses import dataclass

_PATTERN = re.compile(
    r"^(?P<head>.+?) \[(?P<year>\d{4})\] \[(?P<type>[^\]]+)\] "
    r"(?P<format>FLAC|MP3|AAC|[A-Z0-9]+) / (?P<rest>.+?)(?: \[(?P<tracker>[^\]]+)\])?$"
)
_LOG = re.compile(r"^Log \((\d+)%\)$")
_SPECIAL = re.compile(r"deluxe|expanded|special|anniversary|bonus|complete", re.I)
_LEADING_YEAR = re.compile(r"^\d{4}\b\s*")


@dataclass(frozen=True)
class Release:
    name: str
    artist: str
    title: str
    year: int
    kind: str
    encoding: str
    media: str
    log: int | None
    cue: bool
    edition: str

    @property
    def vinyl(self) -> bool:
        return self.media == "Vinyl"

    @property
    def special(self) -> bool:
        return bool(_SPECIAL.search(self.edition))


def parse_release(name: str) -> Release | None:
    m = _PATTERN.match(name.strip())
    if not m:
        return None
    head = m["head"]
    if " - " not in head:
        return None
    artist, title = head.split(" - ", 1)
    parts = [p.strip() for p in m["rest"].split(" / ")]
    encoding = parts[0]
    log = None
    cue = False
    idx = 1
    while idx < len(parts):
        lm = _LOG.match(parts[idx])
        if lm:
            log = int(lm[1])
        elif parts[idx].lower() == "cue":
            cue = True
        else:
            break
        idx += 1
    media = parts[idx] if idx < len(parts) else "Unknown"
    edition = _LEADING_YEAR.sub("", " / ".join(parts[idx + 1 :])).strip()
    return Release(
        name=name,
        artist=artist.strip(),
        title=title.strip(),
        year=int(m["year"]),
        kind=m["type"],
        encoding=encoding,
        media=media,
        log=log,
        cue=cue,
        edition=edition,
    )


MEDIA = ("CD", "SACD", "WEB", "Vinyl")


def _media_rank(r: Release) -> int:
    if r.media == "SACD":
        return 4
    if r.media == "CD" and r.log == 100 and r.cue:
        return 3
    if r.media == "WEB":
        return 2
    if r.media == "CD":
        return 1
    return 0


def quality_key(r: Release, media: str | None = None) -> tuple:
    """Larger is better.

    With no preference vinyl always loses, then special editions, 24bit,
    lossless, and SACD > perfect-log CD > WEB > other CD. A requested `media`
    outranks everything else.
    """
    preferred = int(media is not None and r.media.casefold() == media.casefold())
    return (
        preferred,
        int(not r.vinyl),
        int(r.special),
        int("24bit" in r.encoding.lower()),
        int("lossless" in r.encoding.lower()),
        _media_rank(r),
    )


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def best_versions(names: list[str], media: str | None = None) -> list[Release]:
    groups: dict[tuple, Release] = {}
    for name in names:
        r = parse_release(name)
        if r is None:
            continue
        key = (r.artist.casefold(), _norm(r.title), r.kind)
        current = groups.get(key)
        if current is None or quality_key(r, media) > quality_key(current, media):
            groups[key] = r
    return list(groups.values())
