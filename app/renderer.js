const io = require('socket.io-client');
const { webUtils } = require('electron');
const socket = io('http://localhost:8080');

const dropZone = document.getElementById('drop-zone');
const canvas = document.getElementById('v-canvas');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrub');
const processBtn = document.getElementById('process-btn');
const modeSelect = document.getElementById('mode');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const modelSelect = document.getElementById('model-select');

// --- UI helpers (toast, overlay) ---
function showToast(message, type = 'info', code = null) {
    const host = document.getElementById('toast-host');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    let html = `<div class="toast-msg">${escapeHtml(message)}</div>`;
    if (code) html += `<div class="toast-code">[${escapeHtml(code)}]</div>`;
    el.innerHTML = html;
    host.appendChild(el);
    // Animate in
    requestAnimationFrame(() => el.classList.add('show'));
    // Auto-dismiss after 6s (errors stay longer)
    const ttl = type === 'error' ? 9000 : 5000;
    setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => el.remove(), 300);
    }, ttl);
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

function setLoading(on, label = 'Working…') {
    const overlay = document.getElementById('loading-overlay');
    if (!overlay) return;
    overlay.querySelector('.loading-label').textContent = label;
    overlay.style.display = on ? 'flex' : 'none';
}

function setPhase(label) {
    const el = document.getElementById('phase-label');
    if (el) el.textContent = label;
}

// Tell Python which AI model is selected (SAM2 vs RVM)
modelSelect.addEventListener('change', () => {
    socket.emit('select_model', { model: modelSelect.value });
    if (modelSelect.value === 'rvm') {
        statusText.innerText = 'Auto-Human Mode (No clicks needed)';
    } else {
        statusText.innerText = 'SAM2 Mode (Click to Track)';
    }
});


let currentVideoData = null;
let currentVideoPath = null;
let lastVideoFrameB64 = null; // THIS IS THE CURRENT BACKGROUND LAYER

socket.on('connect', () => {
    statusDot.className = 'dot online';
    statusText.innerText = 'AI Engine Online';
    const banner = document.getElementById('offline-banner');
    if (banner) banner.style.display = 'none';
});

socket.on('disconnect', () => {
    statusDot.className = 'dot';
    statusText.innerText = 'Engine offline';
    const banner = document.getElementById('offline-banner');
    if (banner) banner.style.display = 'flex';
});

socket.on('video_loaded', (data) => {
    setLoading(false);
    currentVideoData = data;
    lastVideoFrameB64 = data.first_frame_b64;
    dropZone.style.display = 'none';
    document.getElementById('canvas-wrap').style.display = 'flex';
    processBtn.disabled = false;
    scrubber.max = data.total_frames - 1;
    scrubber.value = 0;
    document.getElementById('file-info').innerText = `${data.width}x${data.height} | ${data.fps} FPS`;
    renderCanvas(lastVideoFrameB64, null);
});

// --- NEW: HANDLE BACKGROUND FRAME UPDATES ---
socket.on('frame_update', (data) => {
    lastVideoFrameB64 = data.image_b64;
    // When scrubbing, we show the clean frame (mask is null until next click or result)
    renderCanvas(lastVideoFrameB64, null);
});

socket.on('mask_preview', (data) => {
    renderCanvas(lastVideoFrameB64, data.mask_alpha_b64);
});

socket.on('progress_update', (data) => {
    document.getElementById('progress-area').style.display = 'block';
    document.getElementById('progress-bar').style.width = data.percentage + '%';
    document.getElementById('progress-label').innerText = data.message;
});

socket.on('phase_update', (data) => {
    setPhase(data.phase);
    // Keep progress visible during processing phases
    if (data.phase && data.phase !== 'Done' && data.phase !== 'Error') {
        document.getElementById('progress-area').style.display = 'block';
    }
});

socket.on('process_complete', (data) => {
    setPhase('Done');
    setLoading(false);
    showToast(`✨ Export Complete!\n${data.output_files[0]}`, 'success');
    document.getElementById('progress-area').style.display = 'none';
    processBtn.disabled = false;
});

socket.on('error_alert', (data) => {
    setLoading(false);
    const code = data.code || 'ERROR';
    const detail = data.detail ? `\n\n${data.detail}` : '';
    showToast(`${data.message}${detail}`, 'error', code);
    document.getElementById('progress-area').style.display = 'none';
    processBtn.disabled = false;
});

document.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add('active'); });
document.addEventListener('dragleave', () => dropZone.classList.remove('active'));
document.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file) {
        const path = webUtils.getPathForFile(file);
        currentVideoPath = path;
        statusText.innerText = 'Extracting Video...';
        setLoading(true, 'Extracting frames…');
        socket.emit('load_video', { path: path });
    }
});

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('mousedown', (e) => {
    if (!currentVideoData || modelSelect.value === 'rvm') return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const isPositive = (e.button === 0);

    socket.emit('add_click', {
        frame: parseInt(scrubber.value),
        x: x, y: y, is_positive: isPositive
    });
});

processBtn.addEventListener('click', () => {
    if (!currentVideoPath) return;
    const outputDir = currentVideoPath.substring(0, currentVideoPath.lastIndexOf('/'));
    processBtn.disabled = true;
    setPhase('Starting…');
    setLoading(true, 'Processing…');
    socket.emit('start_processing', { format: modeSelect.value, output_dir: outputDir });
});

function renderCanvas(videoB64, maskAlphaB64) {
    if (!videoB64) return;
    const videoImg = new Image();
    videoImg.onload = () => {
        canvas.width = videoImg.width;
        canvas.height = videoImg.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(videoImg, 0, 0);
        if (maskAlphaB64) {
            const maskImg = new Image();
            maskImg.onload = () => { ctx.drawImage(maskImg, 0, 0); };
            maskImg.src = maskAlphaB64;
        }
    };
    videoImg.src = videoB64;
}

// --- NEW: SCRUBBER INTERACTION ---
scrubber.oninput = () => {
    const currentFrame = parseInt(scrubber.value);
    document.getElementById('frame-num').innerText = "Frame: " + currentFrame;
    // Tell Python to send the image for this frame index
    socket.emit('request_frame', { frame: currentFrame });
};
