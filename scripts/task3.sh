#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/task"
sudo python3 task3.py "$@"
