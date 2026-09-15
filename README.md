# Sprintboy

A simple langchain-based tool for adding music to my Plex server.

## Set up using uv

```
uv sync
```

## Usage

```
uv run main.py "Query to send to the agent."
```

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

