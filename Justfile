set dotenv-load

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
    uv run pytest

build:
    docker build -t sprintboy:latest .

deploy: build
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "Building production image..."
    docker save sprintboy:latest | gzip > /tmp/sprintboy.tar.gz
    
    echo "Copying to Unraid..."
    scp /tmp/sprintboy.tar.gz root@shiitake:/tmp/
    scp docker-compose.prod.yml root@shiitake:/mnt/user/appdata/sprintboy/docker-compose.yml
    scp .env.production root@shiitake:/mnt/user/appdata/sprintboy/.env.production
    
    echo "Deploying on Unraid..."
    ssh root@shiitake << 'EOF'
        cd /mnt/user/appdata/sprintboy
        docker load < /tmp/sprintboy.tar.gz
        docker compose down
        docker compose up -d
        rm /tmp/sprintboy.tar.gz
    EOF
    
    rm /tmp/sprintboy.tar.gz
    echo "Deployment complete!"

# Deploy without rebuilding (faster for quick iterations)
deploy-quick:
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "Copying source to Unraid..."
    ssh root@shiitake "mkdir -p /mnt/user/appdata/sprintboy"
    rsync -avz --delete ./src/ root@shiitake:/mnt/user/appdata/sprintboy/src/
    rsync -avz ./pyproject.toml ./uv.lock root@shiitake:/mnt/user/appdata/sprintboy/
    
    echo "Restarting bot..."
    ssh root@shiitake "cd /mnt/user/appdata/sprintboy && docker compose restart"
    
    echo "Quick deploy complete!"

# View logs from Unraid
logs:
    ssh root@shiitake "docker logs -f sprintboy"

# SSH into Unraid bot container
shell:
    ssh -t root@shiitake "docker exec -it sprintboy /bin/bash"

# Check bot status on Unraid
status:
    ssh root@shiitake "docker ps | grep sprintboy"
