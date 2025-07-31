#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/task"           # ← 新增：进入 task 子目录
sudo python3 task1.py "$@"
