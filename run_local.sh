#!/usr/bin/env bash
set -e

# Small helper to create a venv, install dependencies and start the Flask app in background
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv .venv..."
  python3 -m venv .venv
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip and installing requirements (may take a while)..."
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found; please install dependencies manually."
fi

# Start the app in background
echo "Starting Flask app (logs -> app.log). Visit http://<your-ip>:5000 from your phone"
nohup python3 app.py > app.log 2>&1 &

echo "App started. Check app.log for output. To stop: pkill -f 'python3 app.py' or kill <pid>"