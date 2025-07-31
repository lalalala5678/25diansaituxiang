#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/task"           # 进入 task 子目录
sudo python3 task2.py "$@"
