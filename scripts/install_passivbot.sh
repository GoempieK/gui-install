#!/usr/bin/env bash
set -euo pipefail

BOT_ID=""
BOT_ROLE=""
BACKEND_URL=""
BOT_TOKEN=""
BOT_IP=""
WORKDIR=""
PASSIVBOT_REPO="https://github.com/enarjord/passivbot.git"
AGENT_SOURCE="https://raw.githubusercontent.com/GoempieK/passivbot-gui/main/examples/bot_api.py"

usage() {
  cat <<USAGE >&2
Usage: $0 --id <BOT_ID> --role <ROLE> --backend <BACKEND_URL> --token <TOKEN> [--ip <IP_OR_HOSTNAME>] [--workdir <PATH>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --id)
      BOT_ID="$2"
      shift 2
      ;;
    --role)
      BOT_ROLE="$2"
      shift 2
      ;;
    --backend)
      BACKEND_URL="$2"
      shift 2
      ;;
    --token)
      BOT_TOKEN="$2"
      shift 2
      ;;
    --ip)
      BOT_IP="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BOT_ID" || -z "$BOT_ROLE" || -z "$BACKEND_URL" || -z "$BOT_TOKEN" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="/opt/passivbot/${BOT_ID}"
fi

sudo_if_needed() {
  if [[ $EUID -ne 0 ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

ensure_pkg() {
  local pkg="$1"
  if command -v "$pkg" >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    sudo_if_needed apt-get update
    sudo_if_needed apt-get install -y "$pkg"
  elif command -v yum >/dev/null 2>&1; then
    sudo_if_needed yum install -y "$pkg"
  else
    echo "Unable to install $pkg automatically. Please install it and re-run." >&2
    exit 1
  fi
}

ensure_pkg curl
ensure_pkg git

if ! command -v docker >/dev/null 2>&1; then
  echo "Installing Docker..."
  curl -fsSL https://get.docker.com | sudo_if_needed sh
fi

if docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE=(docker-compose)
else
  echo "Docker Compose not found. Attempting installation..."
  sudo_if_needed curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo_if_needed chmod +x /usr/local/bin/docker-compose
  DOCKER_COMPOSE=(docker-compose)
fi

echo "Setting up workspace at $WORKDIR"
sudo_if_needed mkdir -p "$WORKDIR"
sudo_if_needed chown "$USER":"$USER" "$WORKDIR"
cd "$WORKDIR"

if [[ ! -d passivbot ]]; then
  git clone "$PASSIVBOT_REPO" passivbot
else
  (cd passivbot && git pull --ff-only || true)
fi

mkdir -p agent
curl -fsSL "$AGENT_SOURCE" -o agent/bot_api.py

cat > agent/requirements.txt <<'REQ'
fastapi==0.111.0
uvicorn[standard]==0.30.0
httpx==0.27.0
REQ

SELF_URL="http://localhost:9000"
if [[ -n "$BOT_IP" ]]; then
  SELF_URL="http://${BOT_IP}:9000"
fi

export BOT_ID BOT_ROLE BACKEND_URL BOT_TOKEN BOT_IP SELF_URL

cat > agent.env <<EOF
BOT_ID=${BOT_ID}
BOT_ROLE=${BOT_ROLE}
BOT_API_KEY=${BOT_TOKEN}
BACKEND_URL=${BACKEND_URL}
SELF_URL=${SELF_URL}
BOT_IP=${BOT_IP}
BOT_VERSION=latest
EOF

cat > passivbot.env <<EOF
PASSIVBOT_USER=${BOT_ID}
EOF

cat > docker-compose.yml <<'EOF'
version: "3.9"
services:
  passivbot:
    build:
      context: passivbot
      dockerfile: Dockerfile
    container_name: ${BOT_ID}-passivbot
    restart: unless-stopped
    working_dir: /app
    volumes:
      - ./passivbot:/app
    env_file:
      - passivbot.env
    command: ["python", "src/main.py", "configs/template.json"]

  bot_agent:
    image: python:3.11-slim
    container_name: ${BOT_ID}-agent
    working_dir: /agent
    volumes:
      - ./agent:/agent
    env_file:
      - agent.env
    ports:
      - "9000:9000"
    command: >
      sh -c "pip install --no-cache-dir -r requirements.txt && uvicorn bot_api:app --host 0.0.0.0 --port 9000"
    restart: unless-stopped
EOF

${DOCKER_COMPOSE[@]} up -d --build bot_agent

echo "Remote Passivbot agent deployed."
echo "Workspace: $WORKDIR"
echo "Use '${DOCKER_COMPOSE[*]} up -d' to start the full Passivbot stack when ready."
