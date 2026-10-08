"""
Every question set the bot asks a decision model, in one place.

Each set has a versioned name that goes into the decision log. Bump the version
whenever the wording changes, so answers to different wordings are never mixed
when comparing backends or building training data.
"""

ROUTE_QUESTIONS_NAME = "route/v3"

# State: {"message": latest text, "earlier": [{"from": "user"|"bot", "text": ...}]}
# with `earlier` oldest first and absent for a fresh message. v1 saw only the
# message, so a follow-up like "it's on Night Passage but nice try" read as
# chat and the bot lost its download tools mid-conversation.
_CONTEXT = (
    " Judge `message`, the newest message; use `earlier` (the conversation so "
    "far, oldest first) only to work out what it refers to."
)

ROUTE_QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "What kind of media does the message ask for or ask about?" + _CONTEXT,
        "criteria": {
            "music": "albums, artists, songs, genres of music",
            "movie": "films",
            "tv": "TV shows, seasons or episodes",
            "tracker": (
                "their Orpheus tracker account: ratio, bonus points, buying or counting "
                "freeleech tokens (\"tokens\" on its own means these)"
            ),
            "chat": "none in particular: thanks, small talk, or a question about the bot itself",
        },
    },
    "kind": {
        "type": "choice",
        "instructions": "What kind of request is this?" + _CONTEXT,
        "criteria": {
            "specific": (
                "names or points at particular albums, films or episodes to get, "
                "including correcting or retrying an earlier request ('no, the one "
                "with track X', 'it's on Y', 'try the deluxe one')"
            ),
            "discography": "wants an artist's complete or remaining catalogue, or their newest release",
            "open_ended": "leaves the choice to us: recommendations, 'something like X', a genre or mood, 'surprise me', 'more like that'",
            "question": "asks something or chats, and wants nothing downloaded",
        },
    },
}

ACCOUNT_NAME = "account/v1"


def account_questions() -> dict:
    """What an Orpheus account message wants. State: {"message": text}."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "What does the message want to do with the user's tracker account?",
            "criteria": {
                "status": "asks how many tokens / bonus points / what ratio",
                "buy": "wants to buy freeleech tokens with bonus points",
            },
        },
        "max": {
            "type": "noul",
            "instructions": "Does the message ask to buy as many tokens as possible?",
        },
    }


SAME_RELEASE_QUESTIONS_NAME = "same_release/v1"

SAME_RELEASE_QUESTIONS = {
    "same_release": {
        "type": "noul",
        "instructions": (
            "Is `candidate` the same album as `owned`, so that downloading it would "
            "duplicate what the user already has? A deluxe, remastered or expanded "
            "edition of the same album counts as the same album. A different album, "
            "a live recording, a single, or a release by a different artist does not."
        ),
    },
}

DISCOGRAPHY_QUESTIONS_NAME = "discography/v2"


def discography_questions(spans: list[str]) -> dict:
    """Which span of the message names the artist, and which of their releases are wanted."""
    return {
        "artist": {
            "type": "choice",
            "instructions": "Which of these is the name of the artist whose releases the message asks for?",
            "criteria": {span: None for span in spans}
            | {"none": "none of these is an artist name"},
        },
        "scope": {
            "type": "choice",
            "instructions": "Which of the artist's releases does the message want?",
            "criteria": {
                "albums": (
                    "their albums. The default for 'discography', 'the rest of', "
                    "'collection', 'catalogue' or 'everything by'"
                ),
                "everything": (
                    "only when the message explicitly asks for singles, EPs, live "
                    "records or B-sides as well as albums"
                ),
                "newest": "only their newest or latest release",
            },
        },
    }

RECOMMEND_SEED_NAME = "recommend_seed/v1"


def recommend_seed_questions(options: list[str]) -> dict:
    """What an open-ended request is based on: an artist, an album or a tag, and its preferences."""
    return {
        "seed": {
            "type": "choice",
            "instructions": (
                "Which of these is what the recommendations should be based on: an artist, "
                "an album, or a genre, style or mood? Judge `message`; use `earlier` only to "
                "resolve words like 'that' or 'more like it'."
            ),
            "criteria": {option: None for option in options}
            | {"none": "none of these is something to base recommendations on"},
        },
        "seed_type": {
            "type": "choice",
            "instructions": "What kind of thing is the recommendation based on?",
            "criteria": {
                "artist": "a musician or band",
                "album": "a particular album",
                "tag": "a genre, style, scene, era or mood",
                "none": "nothing to base it on ('surprise me')",
            },
        },
        "same_artist": {
            "type": "noul",
            "instructions": "Does the message want more releases by that same artist, rather than other artists?",
        },
        "new_to_them": {
            "type": "noul",
            "instructions": "Does the message want artists the user does not already listen to?",
            "criteria": {
                "true": "new, discover, introduce, haven't heard",
                "false": "no preference",
            },
        },
    }


RECOMMEND_FIT_NAME = "recommend_fit/v1"


def recommend_fit_questions(keys: list[str]) -> dict:
    """How well each candidate album fits the request."""
    return {
        key: {
            "type": "score",
            "instructions": (
                f"How well does `candidates.{key}` fit what `request` asks for, including any "
                f"qualifiers such as heavier, older, from a decade, or a mood?"
            ),
            "criteria": ["does not fit", "loosely fits", "fits", "fits very well"],
        }
        for key in keys
    }

SPECIFIC_EXTRACT_NAME = "specific_extract/v1"


def specific_extract_questions(spans: list[str]) -> dict:
    """Which spans are the artist, a song title and the albums asked for.

    State: the route state plus {"spans": {"album_<i>": span}}, so each noul
    question can point at its span.
    """
    questions: dict = {
        "artist": {
            "type": "choice",
            "instructions": (
                "Which of these is the artist whose release `message` asks for? "
                "Use `earlier` only to resolve references like 'it' or 'that one'."
            ),
            "criteria": {span: None for span in spans}
            | {"none": "none of these is an artist name"},
        },
        "track": {
            "type": "choice",
            "instructions": (
                "Which of these is a song title that `message` names to identify an "
                "album (e.g. 'the album with the track X'), if any?"
            ),
            "criteria": {span: None for span in spans}
            | {"none": "the message names no song"},
        },
    }
    for i in range(len(spans)):
        questions[f"album_{i}"] = {
            "type": "noul",
            "instructions": (
                f"Is `spans.album_{i}` the title of an album (or other release) that "
                f"`message` asks to download? Not the artist's name, not a song."
            ),
        }
    return questions


SPECIFIC_PICK_NAME = "specific_pick/v3"


def specific_pick_questions(keys: list[str]) -> dict:
    """Which candidate release is the one asked for."""
    return {
        "pick": {
            "type": "choice",
            "instructions": (
                "Which of `candidates` is the release `message` asks for? Allow for "
                "misspellings and misremembered titles. A candidate whose `has_track` "
                "is the song they named is very likely it, and when no artist is named "
                "they most likely mean the well-known release (see `artist_listeners`). "
                "'none' if none of them is it."
            ),
            "criteria": {key: None for key in keys}
            | {"none": "none of the candidates is the release asked for"},
        },
    }
