// static/js/params.js
// 1) 刷新界面显示的四个参数
async function refreshParams() {
    const res = await fetch("/api/params");
    const data = await res.json();
    for (const k in data) {
        const el = document.getElementById(k);
        if (el) el.textContent = data[k];
    }
}

document.addEventListener("DOMContentLoaded", () => {
    /* ---------- 初始加载 ---------- */
    refreshParams();

    /* ---------- 加 / 减 按钮 ---------- */
    document.getElementById("params").addEventListener("click", async (e) => {
        const btn = e.target;
        if (!btn.classList.contains("btn")) return; // 只拦截 .inc / .dec

        const block = btn.closest(".param-block");
        const name = block.dataset.name;
        const delta = btn.classList.contains("inc") ? 1 : -1;

        await fetch(`/api/param/${name}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ delta }),
        });

        refreshParams();
    });

    /* ---------- 保存参数按钮 ---------- */
    document.getElementById("save-btn").addEventListener("click", async () => {
        try {
            const res = await fetch("/api/save", { method: "POST" });
            const msg = await res.text();        // 后端返回纯文本
            alert(msg);                          // 简单弹窗提示
        } catch (err) {
            alert("保存失败: " + err);
        }
    });
});
