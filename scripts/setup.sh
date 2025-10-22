#!/usr/bin/env bash
set -e

echo "=== Installing Python dependencies from requirements.txt ==="

apt update
apt install tmux

pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installation complete! ==="