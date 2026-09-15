set dotenv-load
set export

default:
    @just --list

install:
    uv sync

run:
    uv run python -m bot.main

dev:
    docker compose up --build

dev-daemon:
    docker compose up --build -d

dev-stop:
    docker compose down

test:
    uv run pytest -s

# One-shot query through the agent, no discord round trip
ask *QUERY:
    uv run python -m bot.agent_cli {{QUERY}}

# Which models the configured ollama server has, and what is loaded
ollama-models:
    curl -s $OLLAMA_API_URL/api/tags | python -m json.tool
    tailscale ssh root@shiitake "docker exec ollama ollama ps"

# Follow the model server's logs here (add 2>&1 | grep slot for scheduling)
ollama-logs:
    tailscale ssh root@shiitake "docker logs -f --tail 50 ollama"

build:
    docker build -t sprintboy:latest .

deploy: build
    echo "Building production image..."
    docker save sprintboy:latest | gzip > /tmp/sprintboy.tar.gz

    echo "Copying to Unraid and unpacking..."
    cat /tmp/sprintboy.tar.gz | tailscale ssh root@shiitake "cat > /tmp/sprintboy.tar.gz"
    cat docker-compose.prod.yml | tailscale ssh root@shiitake "cat > /mnt/user/appdata/sprintboy/docker-compose.yml"
    cat .env.production | tailscale ssh root@shiitake "cat > /mnt/user/appdata/sprintboy/.env.production"
    tailscale ssh root@shiitake "docker load --input /tmp/sprintboy.tar.gz"

    rm /tmp/sprintboy.tar.gz
    echo "Deployment complete! Go to Unraid to Update"

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
