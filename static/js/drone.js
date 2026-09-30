const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const annotated = document.getElementById("annotated");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const detectionList = document.getElementById("detectionList");
const fpsBadge = document.getElementById("fpsBadge");

let stream = null;
let running = false;
let lastTs = performance.now();
let frames = 0;

async function start() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = "block";
        annotated.style.display = "block";
        video.style.display = "none";
        running = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        loop();
    } catch (e) {
        alert("Camera access denied: " + e.message);
    }
}

function stop() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    startBtn.disabled = false;
    stopBtn.disabled = true;
    annotated.style.display = "none";
}

async function loop() {
    if (!running) return;
    if (video.readyState >= 2) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

        try {
            const res = await fetch("/admin/drone/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl }),
            });
            const data = await res.json();
            annotated.src = data.image;

            detectionList.innerHTML = "";
            if (data.detections.length === 0) {
                detectionList.innerHTML = `<li class="list-group-item text-muted">No persons detected</li>`;
            } else {
                data.detections.forEach(d => {
                    const cls = d.label.toLowerCase() === "injured" ? "danger" : "success";
                    detectionList.innerHTML += `
                        <li class="list-group-item d-flex justify-content-between">
                            <span class="badge bg-${cls}">${d.label}</span>
                            <span>${(d.confidence * 100).toFixed(1)}%</span>
                        </li>`;
                });
            }
        } catch (e) { console.error(e); }
    }
    frames++;
    const now = performance.now();
    if (now - lastTs > 1000) {
        fpsBadge.textContent = frames + " FPS";
        frames = 0;
        lastTs = now;
    }
    requestAnimationFrame(loop);
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);