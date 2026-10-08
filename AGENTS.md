# AGENTS.md

Notes for coding agents (and people) working on sprintboy. Read this before changing the
request flow, the decision models, or anything that touches the unraid box.

## What it is

A Discord bot that adds music, films and TV to a Plex server by grabbing torrents through
qBittorrent (Jackett indexers, mostly Orpheus for music), and answers questions about the
owner's Orpheus account. It runs as a container on an unraid box (`shiitake`, reached with
`tailscale ssh root@shiitake`; LAN/tailnet IP 100.99.141.127).

## Request flow (src/bot)

```
Discord message (main.on_message)
  -> turn.prepare: route with a decision model (routing.route, questions "route/v3")
       state = latest message + last 4 turns ("earlier"), clipped to 300 chars each
  -> turn.run: a deterministic workflow if the route is trusted, else the LLM agent
       tracker            -> workflows.account     (tokens, bonus points, ratio, buy tokens)
       music/specific     -> workflows.specific    (named albums; did-you-mean buttons)
       music/discography  -> workflows.discography (an artist's missing albums)
       music/open_ended   -> workflows.recommend   (seed -> Last.fm candidates -> Jev ranks)
       anything else, or a workflow returning None -> agent.run_agent with select_tools(route)
  -> reply, buttons (choices.py), run log, wait_for_downloads -> Plex scan
```

- Workflows are the fast path (1-7 s); the LLM agent (qwen3:30b-a3b on ollama) takes
  30-100 s and is the fallback. When a workflow is unsure it must return `None`, not guess.
- **No LangChain.** `toolkit.py` (`@tool`, `Tool`, `Runtime`), `llm.py` (`Message`,
  `OllamaChat` over `/api/chat`, `AnthropicChat` over the SDK) and `agent.py` (`run_agent`:
  concurrent tool calls, errors fed back to the model, forced no-tools reply at the step
  cap) are ours. `output.py` reads messages by `.type` (`human`/`ai`/`tool`), `.content`,
  `.tool_calls[i]["name"]`.
- `turn.run` sends one "nudge" when a download turn never tried to add (`output.add_nudge`);
  never after a step cap.
- `releases.py` ranks tracker results in code: SACD > perfect-log CD > WEB > other CD,
  special editions first, never vinyl unless asked; a requested media wins outright.
  `tools.download_many` does library check -> search -> pick -> add for a list of albums.
- Title matching respects volume/part numbers (`tools.release_numbers`): "Djesse Vol. 3"
  is not Vol. 4, "...Upon You ii" is not "...Upon You"; an unnumbered title is only a
  *maybe* for volume 1. Discographies skip maybes rather than risk duplicates.

## Decision models (decide.py, questions.py)

Typed choice/score/noul questions answered with calibrated probabilities over TypeSafe's
`/v1/systemone` wire format.

- **Jev** (hosted, `TYPESAFE_API_KEY`) is primary. ~0.2 s, ~$0.042 per million input
  tokens; a message costs a few hundredths of a cent. It returns the odd
  `503 model_unavailable`; `decide()` retries once.
- **DJ Laya** (`djlaya` on the self-hosted ollaya) is the shadow: asked in the background,
  logged, and used only if Jev fails on a distilled question set (`decide.DISTILLED`:
  `route/v3`, `same_release/v1`).
- Every decision is appended to `data/decisions.jsonl` (state, questions, both backends'
  answers, latency). That is the distillation dataset.
- **Version question sets.** Changing any wording means bumping its name (`route/v2` ->
  `route/v3`) so logged answers to different wordings are never mixed.
- Jev weighs structured state far more than prose: pass candidates as objects
  (`{"artist", "title", "artist_listeners", "has_track"}`), not `"Artist - Title [note]"`
  (In Rainbows went 0.14 -> 0.82 from that change alone). Hand the picker the span as
  written ("in rainbowz", not "rainbowz").
- Keyword guards are fine where the router is blind: bare "tokens"/"ratio" -> tracker.
- Tests never reach a backend: `tests/conftest.py` sets both to `off`. Patch `decide`
  (or the workflow's helpers) in tests that need answers.

## Evals

- `uv run python evals/decision_eval.py jev djlaya laya` — hand-labelled routing,
  follow-up and same-release cases. Answers are cached in `evals/cache/` by request hash,
  so re-runs are free; only new or reworded cases cost anything.
- `just ask "..."` runs one message through the real pipeline from the terminal. Use
  `QBITTORRENT_DRY_RUN=true` and `RUN_LOG_PATH=/tmp/x.jsonl` when testing.

## DJ Laya: distilling Jev into Laya (evals/distill, Justfile `distill-*`)

1. `just distill-data`: qwen writes ~4k requests (with follow-ups); `gen_pairs.py` builds
   ~2.5k album pairs from the real Plex library vs Last.fm (weighted to near-identical).
2. `just distill-label`: Jev labels them with soft probabilities (~$0.15, cached), then a
   stratified 90/10 split; hand-labelled eval messages are excluded.
3. `just distill-train`: `laya-train --loss soft-ce --shuffle-options`, 4 epochs, ~20 min on
   the 3090. **Stops ollama** for the run and restarts it. The training image
   (`evals/distill/docker/Dockerfile`) patches a laya 0.4.0 bug that runs the "before"
   eval with the model still on the CPU.
4. `just distill-publish`: `publish.py` rebuilds `laya:en`'s manifest with the new weights
   and calibration into the self-hosted registry, then ollaya pulls it as `djlaya`.

v1 results: held-out agreement with Jev 0.63 -> 0.93; hand-labelled eval domain 32/35,
kind 32/35, follow-ups 8/8, same_release 11/12 (Jev 30/33/8/9, stock Laya 27/18/3/9).

Gotchas learnt the hard way:
- ollaya's ONNX graphs are weightless: initializers reference the weights blob by file
  name (`sha256-<hex>`) and byte offset. Reusing laya:en's graph silently serves the stock
  weights, and a fine-tuned safetensors header differs in length, so `publish.py`
  retargets every external initializer through its tensor name. Always run the parity
  check (ollaya answers vs `laya.load(checkpoint).predict`) after publishing.
- Ollaya only pulls from registries; ours is the nginx `registry` service in
  `docker-compose.ollaya.yml` (port 11437) serving `/v2/<ns>/<model>/manifests|blobs`.
  Pull as `http://registry/library/djlaya:latest`, then `ollaya cp` to `djlaya`.
- The 3090 hit an NVIDIA GSP firmware crash (`Xid 120`, `RmInitAdapter failed`) right after
  a training run; only a reboot recovered it. If it recurs, consider
  `NVreg_EnableGpuFirmware=0`. With the GPU gone, qwen falls back to the CPU and starves
  ollaya (DJ Laya timed out at 20 s).

## Infrastructure on unraid

| Service | Port | Compose file | Notes |
|---|---|---|---|
| sprintboy | – | docker-compose.prod.yml | `/mnt/user/appdata/sprintboy`; data volume `/app/data` |
| ollama | 11434 | docker-compose.ollama.yml | qwen3:30b-a3b fills ~23 of 24 GB on the 3090 |
| ollaya | 11435 | docker-compose.ollaya.yml | CPU (`OLLAYA_DEVICE=cpu`); `OLLAYA_API_KEY` required |
| ollaya-registry | 11437 | docker-compose.ollaya.yml | static nginx over `/mnt/mycelium/appdata/ollaya-registry` |
| qBittorrent | 4444 | (unraid app) | Jackett on 9117 |

- Model stores live on the pool (`/mnt/mycelium/appdata/...`), not `/mnt/user`: the fuse
  share makes model loads take minutes.
- `just deploy` builds, ships `.env.production` and the image, and recreates the container
  (it also runs `docker image prune -f`). `just ollaya-deploy` writes ollaya's `.env`
  (API key) next to its compose file. Never commit `.env*` (gitignored).
- Logs survive deploys: `data/logs/sprintboy.log` (rotating) and `data/runs.jsonl` (one
  line per turn: route, every tool/workflow step with args and results, reply, timings).
  Read those first when reviewing behaviour; `docker logs` resets on every deploy.

## Orpheus

- `ORPHEUS_API_KEY` (the same key Jackett uses) for `ajax.php`; rate limit 5 req / 10 s,
  enforced in `orpheus.py`. The bonus shop is not in the API: buying tokens posts to
  `bonus.php` with `ORPHEUS_SESSION_COOKIE` and the authkey, only after a Confirm button.
- The API's `ratio` is rounded to two places; use `AccountStats.exact_ratio` and
  `headroom`. The owner runs orpheusbetter to pad the ratio, so dipping near 0.60 is OK.
- Jackett is configured to use freeleech tokens when available, so downloads spend tokens.
- `aotm.py` DMs the owner each Album of the Month (freeleech FLAC, two weeks) with
  Grab/Skip buttons and the cover, linking the public Last.fm page (Orpheus links need a
  logged-in browser). State in `data/aotm.json`.
- Never print or log the API key, authkey, passkey or session cookie; the `index` and
  `user` API responses contain the authkey and passkey.

## Conventions

- Tests: `uv run pytest -q --deselect tests/test_netcode.py::TestPlexAPIClient::test_make_playlist --deselect tests/test_netcode.py::TestQBittorrentClient`
  (those hit live services). They must stay offline and free.
- Comments explain *why*, usually with the incident that motivated the code.
- Be careful with Jev credits in scripts: batch questions per call, cache answers, and
  smoke-test on a handful of rows before a full run.
