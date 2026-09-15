"""
Unit tests for cleaning up model replies before they reach Discord.

The shapes covered here are the ones the self-hosted models actually produced:
leaked think tags, gpt-oss harmony channels, a tool call typed out as prose,
and an empty final message that used to be posted as "Done.".
"""

from bot.output import (
    best_reply,
    describe_failure,
    name_list,
    clean_reply,
    message_text,
    split_for_discord,
    summarise_run,
)


class _AI:
    type = "ai"

    def __init__(self, content):
        self.content = content


class _Tool:
    type = "tool"

    def __init__(self, content):
        self.content = content


def test_plain_text_is_left_alone():
    assert clean_reply("Added Mordechai. You already had Con Todo El Mundo.") == (
        "Added Mordechai. You already had Con Todo El Mundo."
    )


def test_think_block_is_removed():
    content = (
        "<think>The user wants Khruangbin. I should check lastfm first.</think>\n"
        "Added A La Sala."
    )
    assert clean_reply(content) == "Added A La Sala."


def test_unclosed_think_block_drops_the_rest():
    """A cut off response has no answer after the tag, only reasoning."""
    assert clean_reply("Here we go.\n<think>Next I need to check whether") == "Here we go."


def test_orphan_closing_tag_drops_what_came_before():
    """Templates that strip only the opening tag leave the reasoning behind."""
    assert clean_reply("I need to check each album.</think>Added Texas Sun.") == (
        "Added Texas Sun."
    )


def test_harmony_channels_are_reduced_to_the_final_one():
    content = (
        "<|channel|>analysis<|message|>We must check every album one by one.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Added Texas Moon."
    )
    assert clean_reply(content) == "Added Texas Moon."


def test_harmony_analysis_without_a_final_channel_is_dropped():
    content = "<|channel|>analysis<|message|>Thinking about the discography.<|end|>"
    assert clean_reply(content) == ""


def test_tool_call_written_as_prose_is_not_posted():
    content = '{"name": "check_for_album", "arguments": {"artist": "Khruangbin"}}'
    assert clean_reply(content) == ""


def test_json_that_is_actually_an_answer_survives():
    assert clean_reply('{"albums": ["Mordechai"]}') == '{"albums": ["Mordechai"]}'


def test_reasoning_content_blocks_are_dropped():
    content = [
        {"type": "thinking", "thinking": "I should check lastfm."},
        {"type": "text", "text": "Added Mordechai."},
    ]
    assert clean_reply(content) == "Added Mordechai."
    assert message_text(content) == "Added Mordechai."


def test_blank_lines_are_collapsed():
    assert clean_reply("Added one.\n\n\n\nSkipped two.") == "Added one.\n\nSkipped two."


def test_best_reply_falls_back_to_an_earlier_message():
    messages = [
        _AI("Added Mordechai and skipped Con Todo El Mundo."),
        _Tool("COLLISION: the user already owns it."),
        _AI(""),
    ]
    assert best_reply(messages, []) == "Added Mordechai and skipped Con Todo El Mundo."


def test_best_reply_names_the_torrents_when_the_model_says_nothing():
    reply = best_reply([_AI("")], ["Khruangbin - Mordechai [FLAC]"])
    assert reply == "Added:\n- Khruangbin - Mordechai [FLAC]"


def test_best_reply_is_honest_when_nothing_happened():
    reply = best_reply([_AI("<think>hmm</think>")], [])
    assert "nothing was added" in reply
    assert reply != "Done."


def test_short_text_is_one_chunk():
    assert split_for_discord("Added Mordechai.", 2000) == ["Added Mordechai."]


def test_split_breaks_on_line_boundaries():
    lines = [f"- Album number {i}" for i in range(12)]
    chunks = split_for_discord("\n".join(lines), 60)
    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    # No album name may be torn in half across two messages.
    assert sorted(line for chunk in chunks for line in chunk.split("\n")) == sorted(lines)


def test_a_single_overlong_line_is_split_on_a_space():
    chunks = split_for_discord("word " * 40, 50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    # Each chunk is its own discord message, so they rejoin with a separator.
    assert " ".join(chunks).split() == ["word"] * 40


def test_the_real_empty_reply_from_the_deployed_bot():
    """The shape gpt-oss actually produced: every message empty, no tool calls.

    This is the turn that used to be answered with "Done.".
    """
    messages = [
        _AI(""),
        _Tool("Artists similar to Passion Pit:\n- Starfucker (similarity 1.00)"),
        _AI(""),
        _Tool("Albums by Starfucker on Last.fm, most played first:\n- Reptilians"),
        _AI(""),
    ]
    reply = best_reply(messages, [])
    assert reply != "Done."
    assert "nothing was added" in reply


def test_summarise_run_is_one_short_line():
    class _AIWithCalls(_AI):
        def __init__(self, content, calls):
            super().__init__(content)
            self.tool_calls = [{"name": name} for name in calls]

    messages = [
        _AIWithCalls("", ["lastfm_similar_artists"]),
        _Tool("Artists similar to Passion Pit: ..."),
        _AIWithCalls("", ["check_for_album"]),
        _Tool("COLLISION: ..."),
        _AI("Added Reptilians. You already had Jupiter."),
    ]
    line = summarise_run(messages)
    assert "\n" not in line
    assert "lastfm_similar_artists" in line and "check_for_album" in line
    assert "Added Reptilians" in line
    # The system prompt and tool results stay out of it.
    assert "COLLISION" not in line and len(line) < 300


def test_name_list_bullets_every_name_including_the_first():
    """The download notice used to leave the first item without a bullet."""
    assert name_list(["Album A", "Album B"]) == "- Album A\n- Album B"


def test_known_ollama_failures_get_a_useful_explanation():
    empty = describe_failure(ValueError("No data received from Ollama stream."))
    assert "reloaded" in empty and "Ollama stream" not in empty

    garbled = describe_failure(
        Exception("error parsing tool call: raw='We need to...' (status code: 500)")
    )
    assert "garbled" in garbled and "status code" not in garbled


def test_unknown_failures_still_show_the_detail():
    assert describe_failure(RuntimeError("plex is down")) == (
        "Something went wrong: plex is down"
    )
