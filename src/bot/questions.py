"""
Every question set the bot asks a decision model, in one place.

Each set has a versioned name that goes into the decision log. Bump the version
whenever the wording changes, so answers to different wordings are never mixed
when comparing backends or building training data.
"""

ROUTE_QUESTIONS_NAME = "route/v1"

ROUTE_QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "What kind of media does the message ask for or ask about?",
        "criteria": {
            "music": "albums, artists, songs, genres of music",
            "movie": "films",
            "tv": "TV shows, seasons or episodes",
            "chat": "none in particular: thanks, small talk, or a question about the bot itself",
        },
    },
    "kind": {
        "type": "choice",
        "instructions": "What kind of request is this?",
        "criteria": {
            "specific": "names particular albums, films or episodes to get",
            "discography": "wants an artist's complete or remaining catalogue, or their newest release",
            "open_ended": "leaves the choice to us: recommendations, 'something like X', a genre or mood, 'surprise me'",
            "question": "asks something or chats, no download wanted",
        },
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
