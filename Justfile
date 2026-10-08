set dotenv-load
set export

default:
    @just --list

install:
    uv sync

run:
    uv run python -m bot.main

dev:
    GIT_SHA=$(git rev-parse --short HEAD) docker compose up --build

dev-daemon:
    GIT_SHA=$(git rev-parse --short HEAD) docker compose up --build -d

dev-stop:
    docker compose down

test:
    uv run pytest -s

# What the Plex tests left behind. Add --yes to actually delete it.
clean-test-playlists *ARGS:
    uv run python -m bot.plex_cli {{ARGS}}

# One-shot query through the agent, no discord round trip
ask *QUERY:
    uv run python -m bot.agent_cli {{QUERY}}

# What Orpheus' Album of the Month is right now, and the torrent the bot would offer
aotm:
    uv run python -m bot.aotm

# Which models the configured ollama server has, and what is loaded
ollama-models:
    curl -s $OLLAMA_API_URL/api/tags | python -m json.tool
    tailscale ssh root@shiitake "docker exec ollama ollama ps"

# Follow the model server's logs here (add 2>&1 | grep slot for scheduling)
ollama-logs:
    tailscale ssh root@shiitake "docker logs -f --tail 50 ollama"

# Push the decision-model server's compose file to unraid and (re)start it
ollaya-deploy:
    tailscale ssh root@shiitake "mkdir -p /mnt/user/appdata/ollaya /mnt/mycelium/appdata/ollaya /mnt/mycelium/appdata/ollaya-registry/v2 && chown 1000:1000 /mnt/mycelium/appdata/ollaya"
    cat docker-compose.ollaya.yml | tailscale ssh root@shiitake "cat > /mnt/user/appdata/ollaya/docker-compose.yml"
    echo "OLLAYA_API_KEY=$OLLAYA_API_KEY" | tailscale ssh root@shiitake "umask 077 && cat > /mnt/user/appdata/ollaya/.env"
    tailscale ssh root@shiitake "cd /mnt/user/appdata/ollaya && docker compose pull && docker compose up -d"

# Pull a decision model onto the server, e.g. `just ollaya-pull laya`
ollaya-pull MODEL:
    tailscale ssh root@shiitake "docker exec ollaya ollaya pull {{MODEL}}"

# What the decision-model server has, and what is loaded
ollaya-models:
    tailscale ssh root@shiitake "docker exec ollaya ollaya list && docker exec ollaya ollaya ps"

ollaya-logs:
    tailscale ssh root@shiitake "docker logs -f --tail 50 ollaya"

# The commit is baked in so /ping and the startup log can name what is running
build:
    docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t sprintboy:latest .

deploy: build
    echo "Building production image..."
    docker save sprintboy:latest | gzip > /tmp/sprintboy.tar.gz

    echo "Copying to Unraid and unpacking..."
    cat /tmp/sprintboy.tar.gz | tailscale ssh root@shiitake "cat > /tmp/sprintboy.tar.gz"
    cat docker-compose.prod.yml | tailscale ssh root@shiitake "cat > /mnt/user/appdata/sprintboy/docker-compose.yml"
    cat .env.production | tailscale ssh root@shiitake "cat > /mnt/user/appdata/sprintboy/.env.production"
    tailscale ssh root@shiitake "docker load --input /tmp/sprintboy.tar.gz"

    rm /tmp/sprintboy.tar.gz
    echo "Recreating the container on the new image..."
    tailscale ssh root@shiitake "cd /mnt/user/appdata/sprintboy && docker compose up -d && docker image prune -f"
    echo "Deployment complete!"

# Deploy without rebuilding (faster for quick iterations)
deploy-quick:
    echo "Copying source to Unraid..."
    tailscale ssh root@shiitake "mkdir -p /mnt/user/appdata/sprintboy"
    tar czf - ./src | tailscale ssh root@shiitake "cd /mnt/user/appdata/sprintboy && rm -rf src && tar xzf -"
    cat pyproject.toml | tailscale ssh root@shiitake "cat > /mnt/user/appdata/sprintboy/pyproject.toml"
    cat uv.lock | tailscale ssh root@shiitake "cat > /mnt/user/appdata/sprintboy/uv.lock"

    echo "Restarting bot..."
    tailscale ssh root@shiitake "cd /mnt/user/appdata/sprintboy && docker compose restart"

    echo "Quick deploy complete!"

# Follow the deployed bot's logs here
logs:
    tailscale ssh root@shiitake "docker logs -f --tail 100 sprintboy"

# Search the deployed bot's logs, e.g. just logs-grep "Agent response"
logs-grep PATTERN:
    tailscale ssh root@shiitake "docker logs --tail 20000 sprintboy 2>&1" | grep -i --color=always {{PATTERN}}

# Copy the deployed bot's whole log history here to dig through
logs-save FILE="/tmp/sprintboy-deployed.log":
    tailscale ssh root@shiitake "docker logs sprintboy 2>&1" > {{FILE}}
    wc -l {{FILE}}

# SSH into Unraid bot container
shell:
    tailscale ssh root@shiitake "docker exec -it sprintboy /bin/bash"

# Check bot status on Unraid
status:
    tailscale ssh root@shiitake "docker ps | grep sprintboy"

# --- DJ Laya: distil Jev into a local Laya (evals/distill) ---------------------
# 1. Messages from the local LLM and album pairs from the library (free)
distill-data BATCHES="230":
    uv run python evals/distill/gen_messages.py --batches {{BATCHES}} --out data/distill/messages.jsonl
    uv run python evals/distill/gen_pairs.py --out data/distill/pairs.jsonl

# 2. Jev labels (about $0.15 for ~6.5k rows; cached, so re-runs are free) and the train/eval split
distill-label:
    uv run python evals/distill/label.py route data/distill/messages.jsonl data/distill/route.jsonl
    uv run python evals/distill/label.py same_release data/distill/pairs.jsonl data/distill/same_release.jsonl
    uv run python evals/distill/prepare.py

# 3. Train on the 3090. Stops ollama for the ~20 minutes it takes, then restarts it.
distill-train VERSION="v1":
    cat evals/distill/docker/Dockerfile | tailscale ssh root@shiitake "mkdir -p /mnt/user/appdata/djlaya-build && cat > /mnt/user/appdata/djlaya-build/Dockerfile"
    tailscale ssh root@shiitake "cd /mnt/user/appdata/djlaya-build && docker build -q -t djlaya-train ."
    cat data/distill/train.jsonl | tailscale ssh root@shiitake "cat > /mnt/mycelium/appdata/djlaya/train.jsonl"
    cat data/distill/eval.jsonl | tailscale ssh root@shiitake "cat > /mnt/mycelium/appdata/djlaya/eval.jsonl"
    tailscale ssh root@shiitake "docker stop ollama; docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size 8g -v /mnt/mycelium/appdata/djlaya:/work djlaya-train laya-train --data /work/train.jsonl --eval /work/eval.jsonl --base english --out /work/djlaya-{{VERSION}} --loss soft-ce --shuffle-options --epochs 4 --micro-batch 8 --grad-accum 8 --device cuda; docker start ollama"

# 4. Publish to the self-hosted registry and (re)pull it into ollaya as `djlaya`
distill-publish VERSION="v1":
    cat evals/distill/publish.py | tailscale ssh root@shiitake "cat > /mnt/mycelium/appdata/djlaya/publish.py"
    tailscale ssh root@shiitake "docker run --rm -v /mnt/mycelium/appdata/djlaya:/work -v /mnt/mycelium/appdata/ollaya:/ollaya:ro -v /mnt/mycelium/appdata/ollaya-registry:/registry djlaya-train python /work/publish.py /work/djlaya-{{VERSION}} djlaya {{VERSION}}"
    tailscale ssh root@shiitake "docker exec ollaya ollaya pull http://registry/library/djlaya:latest && docker exec ollaya ollaya cp http://registry/library/djlaya:latest djlaya && docker exec ollaya ollaya stop djlaya"
    uv run python evals/decision_eval.py jev djlaya
