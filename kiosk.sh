#!/bin/bash
URL="http://localhost:5000"

rm -f /tmp/.X1-lock /tmp/.X0-lock   # 防止残留锁文件

# 等 Flask 最多 30 秒
for i in {1..30}; do
  if curl -fs "$URL" >/dev/null; then break; fi
  sleep 1
done

# ① 后台启动 X + Chromium 到 :1 vt2
/usr/bin/xinit /usr/bin/chromium-browser \
  --no-sandbox --incognito --kiosk "$URL" \
  -- :1 vt2 -nolisten tcp &

X_PID=$!    # 记录后台进程 PID

# ② 给 X 一点启动时间 → 切换到 vt2
sleep 2
chvt 2

# ③ 等 X / Chromium 退出后脚本返回，方便 systemd 监控
wait $X_PID
