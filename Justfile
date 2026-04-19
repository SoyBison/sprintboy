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

# View logs from Unraid
logs:
    tailscale ssh root@shiitake "docker logs -f sprintboy"

# SSH into Unraid bot container
shell:
    tailscale ssh root@shiitake "docker exec -it sprintboy /bin/bash"

# Check bot status on Unraid
status:
    tailscale ssh root@shiitake "docker ps | grep sprintboy"
