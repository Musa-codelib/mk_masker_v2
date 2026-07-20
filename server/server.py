"""Mk Masker backend server.

A SocketIO (aiohttp) server that bridges the Electron frontend and the Python AI
engines. The frontend emits events (load_video, select_model, add_click,
start_processing, ...) and the server replies with status / frame / mask / progress
events. Heavy CPU/GPU work is pushed to a thread pool via run_in_executor so the
async event loop stays responsive.
"""

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path: sys.path.insert(0, str(current_dir))

import socketio, asyncio, os, shutil, numpy as np, base64, io, logging, re, cv2
from aiohttp import web
from PIL import Image

from core.sam2_runner import SAM2ImageAnnotator, SAM2VideoRunner
from core.rvm_runner import RVMRunner
from utils.paths import resolve_rvm_checkpoint
from utils.video_handler import extract_frames, get_frame_base64
from utils.export import ExportPipeline
from utils.errors import as_mk_error, NoSelectionError
from scripts.download_models import download_model, list_models, MODELS

logging.basicConfig(level=logging.INFO)
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application(); sio.attach(app)

# --- GLOBAL STATE ---
# SESSION: per-video metadata for the currently loaded clip.
# CLICK_MEMORY: user click prompts per frame (SAM2 only).
# SELECTED_MODEL: "sam2" or "rvm", toggled from the UI.
# masks_dir: where per-frame mask PNGs are written between preview and export.
SESSION = {}
CLICK_MEMORY = {"points": {}, "labels": {}}
SELECTED_MODEL = "sam2"
SELECTED_SAM2_VARIANT = "small"
masks_dir = Path.home() / "Library/Caches/com.mkmasker.pro/masks"
# Engine singletons, built lazily on first use (models are heavy to load).
sam2_ann, sam2_run, rvm_run = None, None, None

def _get_ai():
    """Return the (annotator, runner) pair for the currently selected model.

    Builds the engine on first request and caches it so we don't reload weights on
    every interaction. For RVM the annotator is None (RVM needs no clicks).
    """
    global sam2_ann, sam2_run, rvm_run, SELECTED_SAM2_VARIANT
    if SELECTED_MODEL == "sam2":
        if sam2_ann is None or sam2_run is None:
            sam2_ann = SAM2ImageAnnotator(variant=SELECTED_SAM2_VARIANT)
            sam2_run = SAM2VideoRunner(variant=SELECTED_SAM2_VARIANT)
        return sam2_ann, sam2_run
    else:
        if rvm_run is None:
            rvm_run = RVMRunner(resolve_rvm_checkpoint())
        return None, rvm_run

@sio.event
async def connect(sid, environ): logging.info(f"🟢 Connected: {sid}")

@sio.on('load_video')
async def handle_load_video(sid, data):
    """Decode the dropped video into frames and tell the UI it's ready to scrub."""
    global SESSION, CLICK_MEMORY
    await sio.emit('system_status', {'status': 'Extracting Frames...'}, to=sid)
    loop = asyncio.get_running_loop()
    # Frame extraction is CPU-heavy, so run it off the event loop.
    total_f, fps, w, h, scale = await loop.run_in_executor(None, extract_frames, data['path'])
    SESSION = {'path': data['path'], 'total_frames': total_f, 'fps': fps, 'width': w, 'height': h, 'scale': scale,
               'frames_dir': Path.home() / "Library/Caches/com.mkmasker.pro/temp_frames"}
    # Reset per-video state (clicks + any leftover masks).
    CLICK_MEMORY = {"points": {}, "labels": {}}
    if masks_dir.exists(): shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    ai_ann, _ = _get_ai()
    if ai_ann: ai_ann.clear_cache()
    await sio.emit('video_loaded', {'total_frames': total_f, 'fps': fps, 'width': w, 'height': h, 'first_frame_b64': get_frame_base64(0)}, to=sid)

@sio.on('select_model')
async def handle_select_model(sid, data):
    """Switch between SAM2 (click) and RVM (auto) engines."""
    global SELECTED_MODEL, SELECTED_SAM2_VARIANT, sam2_ann, sam2_run
    model = data.get('model', 'sam2-small')
    if model == 'rvm':
        SELECTED_MODEL = 'rvm'
    else:
        SELECTED_MODEL = 'sam2'
        if model == 'sam2-tiny':
            SELECTED_SAM2_VARIANT = 'tiny'
        else:
            SELECTED_SAM2_VARIANT = 'small'
        sam2_ann, sam2_run = None, None

@sio.on('request_frame')
async def handle_request_frame(sid, data):
    """Return a single decoded frame (used when scrubbing the timeline)."""
    await sio.emit('frame_update', {'frame': data['frame'], 'image_b64': get_frame_base64(data['frame'])}, to=sid)

@sio.on('add_click')
async def handle_add_click(sid, data):
    """SAM2 only: record a click prompt and return the resulting mask overlay."""
    if SELECTED_MODEL == "rvm": return  # RVM needs no clicks
    global CLICK_MEMORY
    ai_ann, _ = _get_ai(); f_idx = data['frame']
    # Store the click in full-resolution coords (UI coords * downscale factor).
    CLICK_MEMORY["points"].setdefault(f_idx, []).append([data['x'] * SESSION['scale'], data['y'] * SESSION['scale']])
    CLICK_MEMORY["labels"].setdefault(f_idx, []).append(1 if data['is_positive'] else 0)
    img_path = SESSION['frames_dir'] / f"{f_idx:08d}.jpg"
    ai_ann.set_image(np.array(Image.open(img_path).convert("RGB")), f_idx)
    mask_pil = ai_ann.predict(CLICK_MEMORY["points"][f_idx], CLICK_MEMORY["labels"][f_idx])
    mask_pil.save(masks_dir / f"mask_{f_idx:04d}.png")

    # Build a translucent-blue RGBA overlay so the user sees the current selection.
    full_res_m = mask_pil.resize((SESSION['width'], SESSION['height']), Image.NEAREST); m_np = np.array(full_res_m)
    rgba = np.zeros((m_np.shape[0], m_np.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 2] = 255; rgba[:, :, 3] = (m_np > 0) * 150
    buf = io.BytesIO(); Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    await sio.emit('mask_preview', {'frame': f_idx, 'mask_alpha_b64': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"}, to=sid)

@sio.on('start_processing')
async def handle_start_processing(sid, data):
    """Main job: run the selected engine, then export the masked video.

    Two phases:
      1. Mask generation  - RVM processes every frame, or SAM2 propagates from clicks.
      2. Export           - ExportPipeline turns masks into the final video (matte ->
                             RGBA -> ffmpeg encode) for the chosen format.
    Progress/phase callbacks feed the UI's progress bar and phase label.
    """
    global CLICK_MEMORY, SELECTED_MODEL
    _, ai_run = _get_ai(); main_loop = asyncio.get_running_loop(); mode = data.get('format', 'prores')

    # --- UI callbacks (run on the async loop from worker threads) ---
    def on_prog(f, t):
        # Tracking progress is reported as a % of total frames.
        p = int((f/SESSION['total_frames'])*100)
        asyncio.run_coroutine_threadsafe(sio.emit('progress_update', {'percentage': p, 'message': f"Tracking... {f}/{SESSION['total_frames']}"}, to=sid), main_loop)

    def on_phase(label):
        asyncio.run_coroutine_threadsafe(sio.emit('phase_update', {'phase': label}, to=sid), main_loop)

    def on_export_prog(f, t):
        p = int((f/t)*100) if t else 0
        asyncio.run_coroutine_threadsafe(sio.emit('progress_update', {'percentage': p, 'message': f"{label_hint} {f}/{t}"}, to=sid), main_loop)

    label_hint = "Rendering"  # updated per phase below

    try:
        frame_files = sorted(list(SESSION['frames_dir'].glob("*.jpg")))
        if SELECTED_MODEL == "rvm":
            # RVM: automatic, no clicks. Process every frame in order.
            ai_run.reset()
            for i, f_path in enumerate(frame_files):
                mask_np = ai_run.process_frame(cv2.imread(str(f_path)))
                Image.fromarray(mask_np, mode="L").save(masks_dir / f"mask_{i:04d}.png")
                if i % 5 == 0: on_prog(i, len(frame_files))
        else:
            # SAM2: must have at least one user click to seed tracking.
            if not CLICK_MEMORY["points"]: raise NoSelectionError(
                "No selections found. Click on the subject in the video before processing.")

            # Drop the live preview masks so the export starts from a clean set.
            for f in masks_dir.glob("mask_*.png"): f.unlink()

            # Propagate masks across the whole video (runs in a worker thread).
            await main_loop.run_in_executor(None, ai_run.run_bidirectional, frame_files, CLICK_MEMORY["points"], CLICK_MEMORY["labels"], masks_dir, on_prog)

        # --- Export Pipeline ---
        out_path = Path(data['output_dir']) / f"cutout_{Path(SESSION['path']).stem}"
        render_temp = Path.home() / "Library/Caches/com.mkmasker.pro/render_temp"
        if render_temp.exists(): shutil.rmtree(render_temp)
        render_temp.mkdir(parents=True)

        pipeline = ExportPipeline()
        label_hint = "Rendering alpha masks"
        pipeline.render_matte_frames(masks_dir, render_temp, on_progress=on_export_prog, on_phase=on_phase)
        if mode in ["prores", "balanced"]:
            # These formats need the full RGBA (color + alpha) frames.
            label_hint = "Preparing RGBA frames"
            pipeline.prepare_rgba_frames(SESSION['frames_dir'], render_temp, render_temp, SESSION['total_frames'], on_progress=on_export_prog, on_phase=on_phase)
        label_hint = f"Encoding {mode}"
        final_file = pipeline.encode_video(render_temp, out_path, SESSION['fps'], mode, on_phase=on_phase)
        await sio.emit('process_complete', {'output_files': [str(final_file)]}, to=sid)
        await sio.emit('phase_update', {'phase': 'Done'}, to=sid)
        shutil.rmtree(render_temp)

    except Exception as e:
        # Normalise any failure into a typed payload the UI can show + relay to an editor.
        err = as_mk_error(e)
        logging.error(f"Render Error [{err.code}]: {err.user_message}")
        if err.detail:
            logging.error(f"Detail: {err.detail}")
        await sio.emit('error_alert', err.to_payload(), to=sid)
        await sio.emit('phase_update', {'phase': 'Error'}, to=sid)

@sio.on('get_model_status')
async def handle_get_model_status(sid):
    """Return download status for all SAM2 models."""
    try:
        models = list_models()
        await sio.emit('model_status', {'models': models}, to=sid)
    except Exception as e:
        await sio.emit('error_alert', {
            'code': 'MODEL_STATUS_ERROR',
            'message': str(e),
            'detail': None
        }, to=sid)

@sio.on('download_model')
async def handle_download_model(sid, data):
    """Download a SAM2 model checkpoint."""
    variant = data.get('variant')
    if not variant or variant not in MODELS:
        await sio.emit('error_alert', {
            'code': 'INVALID_MODEL',
            'message': f"Invalid model variant: {variant}",
            'detail': f"Valid variants: {list(MODELS.keys())}"
        }, to=sid)
        return

    try:
        await sio.emit('model_download_start', {'variant': variant}, to=sid)
        loop = asyncio.get_running_loop()

        def on_progress(downloaded, total):
            pct = int(downloaded * 100 / total) if total > 0 else 0
            asyncio.run_coroutine_threadsafe(
                sio.emit('model_download_progress', {
                    'variant': variant,
                    'percentage': pct,
                    'downloaded': downloaded,
                    'total': total
                }, to=sid),
                loop
            )

        path = await loop.run_in_executor(None, download_model, variant, on_progress)
        await sio.emit('model_download_complete', {
            'variant': variant,
            'path': str(path)
        }, to=sid)
    except Exception as e:
        await sio.emit('model_download_error', {
            'variant': variant,
            'error': str(e)
        }, to=sid)

if __name__ == '__main__': web.run_app(app, port=8080)