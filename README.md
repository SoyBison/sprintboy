# Sprintboy

A simple tool for adding music to my Plex server.

## Set up using uv

```
uv sync
```

## Usage

```
just run                                  # the discord bot
just ask "Query to send to the agent."    # one query, straight to the terminal
```

## Choosing a model

`LLM_PROVIDER` picks the backend: `anthropic` for the API, `ollama` for a self-hosted
model. `just ask -p ollama -m qwen3:30b-a3b "..."` overrides it for a single query, which
is the quick way to compare models.

For ollama, set `OLLAMA_API_URL` to the server and `OLLAMA_MODEL` to a model with tool
support — the agent is useless without it. `OLLAMA_NUM_CTX` defaults to 32768 because
ollama's own default of 4096 is smaller than the system prompt plus the nine tool schemas.

Model notes from testing the full pipeline on the unraid box: `qwen3:30b-a3b` completed
every run cleanly at 1.5-3 min. `gpt-oss:20b` is 3-4x faster but failed roughly one run in
three, either emitting malformed tool-call JSON (ollama answers `500 error parsing tool
call`) or finishing with an empty message, so it is not a good default.

The model server itself is `docker-compose.ollama.yml`, which needs the nvidia runtime to
use a GPU. Keep the models on an SSD and mount the pool path directly: read through
unraid's `/mnt/user` fuse share off an array disk, a 13GB model takes minutes to load and
the client gives up before the first token.

## Features

- sprintboy can search for music available in QBittorrent Clients in your network.
    - You can customize the search plugins in that client to add more sources or use private sources.

- Sprintboy can check your plex server to make sure its not downloading the same album just in a different format.

- Sprintboy uses Last.fm for artist and album metadata, so recommendations and discographies
  come from a live database rather than the model's training data.
    - Artist and album names are canonicalised through Last.fm before the Plex check, so a
      request for "Guns and Roses" still matches the "Guns N' Roses" already in your library.
    - Set `LASTFM_API_KEY` in `.env` to enable it (get a key at
      https://www.last.fm/api/account/create). Without it the bot still works, but it falls
      back to the model's own knowledge and to exact-name matching against Plex.

## Use-Cases

- Sprintboy excels over a simple torrent search because it can interpret vague queries.

```
uv run main.py "Get me Metallica's entire discography"
# In this case the agent will check your Plex server to see if any of the albums are already downloaded. Then search for them in QBittorrent, and send to the server.

uv run main.py "Introduce me to new music in the future jazz style."
# In this case the agent will check your plex server for artists who use that style, then recommend more music from similar artists.
```

## Planned Features

- Bandcamp integration to help you find independent artists.

