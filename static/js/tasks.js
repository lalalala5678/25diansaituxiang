// 前端逻辑：执行任务并显示输出
document.addEventListener('DOMContentLoaded', () => {
    const output = document.getElementById('output');

    document.getElementById('tasks').addEventListener('click', async (e) => {
        const btn = e.target;
        if (!btn.classList.contains('task-btn')) return;

        const task = btn.dataset.task;
        btn.disabled = true;
        btn.textContent = `${task}…`;

        try {
            const res = await fetch(`/api/run/${task}`, { method: 'POST' });
            const data = await res.json();
            output.textContent =
                `$ ${task}\n\n${data.stdout || ''}${data.stderr || ''}\n(exit ${data.exit_code})`;
        } catch (err) {
            output.textContent = 'Error: ' + err;
        } finally {
            btn.disabled = false;
            btn.textContent = task;
        }
    });
});
