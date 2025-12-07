// --- Remove loading screen once video loads ---
window.addEventListener("load", () => {
    const app = document.getElementById("app");
    const loading = document.getElementById("loading-screen");

    if (loading && app) {
        setTimeout(() => {
            loading.style.opacity = "0";
            setTimeout(() => loading.remove(), 300);

            app.classList.remove("hidden");
            app.style.opacity = "1";
        }, 600);
    }
});


// --- Live Snail Count Updater ---
function fetchSnailCount() {
    fetch("/snail_count")
        .then(res => res.json())
        .then(data => {
            const el = document.getElementById("snail-count");
            if (el) {
                el.innerText = data.count;
            }
        })
        .catch(err => console.log("Count update error:", err));
}

// Poll every 300ms
window.addEventListener("DOMContentLoaded", () => {
    setInterval(fetchSnailCount, 300);
});
