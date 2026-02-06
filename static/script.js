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
                // Prefer live count for the dashboard main display; fall back to total or legacy fields
                el.innerText = (data.live ?? data.total ?? data.count ?? 0);
            }
            // Update statistics current values
            updateStatsCurrent(data);
        })
        .catch(err => console.log("Count update error:", err));
}

// Update current statistics values
function updateStatsCurrent(data) {
    const totalEl = document.getElementById("stat-total");
    const liveEl = document.getElementById("stat-live");
    if (totalEl) totalEl.textContent = data.total || 0;
    if (liveEl) liveEl.textContent = data.live || 0;
}

// --- Statistics Dropdown ---
let countChart = null;
let fpsChart = null;
let statsExpanded = false;

function initStats() {
    const toggle = document.getElementById("stats-toggle");
    const header = document.getElementById("stats-header");
    const charts = document.getElementById("stats-charts");
    
    if (toggle && header && charts) {
        header.addEventListener("click", () => {
            statsExpanded = !statsExpanded;
            toggle.classList.toggle("expanded", statsExpanded);
            charts.classList.toggle("visible", statsExpanded);
            
            if (statsExpanded && !countChart) {
                initCharts();
            }
            
            if (statsExpanded) {
                updateCharts();
            }
        });
    }
}

// Initialize charts
function initCharts() {
    const countCtx = document.getElementById("countChart");
    const fpsCtx = document.getElementById("fpsChart");
    
    if (countCtx) {
        countChart = new Chart(countCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Total Count',
                        data: [],
                        borderColor: 'rgb(58, 75, 255)',
                        backgroundColor: 'rgba(58, 75, 255, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Live Count',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Snail Count Over Time (Last Minute)'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Seconds Ago'
                        },
                        reverse: true
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Count'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    if (fpsCtx) {
        fpsChart = new Chart(fpsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Inference FPS',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Stream FPS',
                        data: [],
                        borderColor: 'rgb(255, 159, 64)',
                        backgroundColor: 'rgba(255, 159, 64, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'FPS Over Time (Last Minute)'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Seconds Ago'
                        },
                        reverse: true
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'FPS'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Update charts with historical data
function updateCharts() {
    fetch("/stats_history")
        .then(res => res.json())
        .then(data => {
            if (!data || data.length === 0) return;
            
            // Sort by time_ago (ascending, so most recent is last)
            data.sort((a, b) => a.time_ago - b.time_ago);
            
            const labels = data.map(d => d.time_ago);
            
            if (countChart) {
                countChart.data.labels = labels;
                countChart.data.datasets[0].data = data.map(d => d.total_count);
                countChart.data.datasets[1].data = data.map(d => d.live_count);
                countChart.update('none'); // 'none' mode for smooth updates
            }
            
            if (fpsChart) {
                fpsChart.data.labels = labels;
                fpsChart.data.datasets[0].data = data.map(d => d.inference_fps);
                fpsChart.data.datasets[1].data = data.map(d => d.stream_fps);
                fpsChart.update('none');
            }
        })
        .catch(err => console.log("Stats history error:", err));
}

// Fetch FPS data for current stats
function fetchFPSData() {
    fetch("/stats_history")
        .then(res => res.json())
        .then(data => {
            if (data && data.length > 0) {
                const latest = data[data.length - 1];
                const infFpsEl = document.getElementById("stat-inf-fps");
                const strFpsEl = document.getElementById("stat-str-fps");
                if (infFpsEl) infFpsEl.textContent = latest.inference_fps.toFixed(1);
                if (strFpsEl) strFpsEl.textContent = latest.stream_fps.toFixed(1);
            }
        })
        .catch(err => console.log("FPS data error:", err));
}

// Poll every 300ms
window.addEventListener("DOMContentLoaded", () => {
    setInterval(fetchSnailCount, 300);
    setInterval(fetchFPSData, 1000); // Update FPS every second
    setInterval(() => {
        if (statsExpanded) {
            updateCharts();
        }
    }, 2000); // Update charts every 2 seconds when expanded
    
    initStats();
});
