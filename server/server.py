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

logging.basicConfig(level=logging.INFO)
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application(); sio.attach(app)

# --- GLOBAL STATE ---
SESSION = {}
CLICK_MEMORY = {"points": {}, "labels": {}} 
SELECTED_MODEL = "sam2"
masks_dir = Path.home() / "Library/Caches/com.mkmasker.pro/masks"
sam2_ann, sam2_run, rvm_run = None, None, None

def _get_ai():
    global sam2_ann, sam2_run, rvm_run
    if SELECTED_MODEL == "sam2":
        if sam2_ann is None:
            sam2_ann = SAM2ImageAnnotator(variant="small")
            sam2_run = SAM2VideoRunner(variant="small")
        return sam2_ann, sam2_run
    else:
        if rvm_run is None: 
            rvm_run = RVMRunner(resolve_rvm_checkpoint())
        return None, rvm_run

@sio.event
async def connect(sid, environ): logging.info(f"🟢 Connected: {sid}")

@sio.on('load_video')
async def handle_load_video(sid, data):
    global SESSION, CLICK_MEMORY
    await sio.emit('system_status', {'status': 'Extracting Frames...'}, to=sid)
    loop = asyncio.get_running_loop()
    total_f, fps, w, h, scale = await loop.run_in_executor(None, extract_frames, data['path'])
    SESSION = {'path': data['path'], 'total_frames': total_f, 'fps': fps, 'width': w, 'height': h, 'scale': scale, 
               'frames_dir': Path.home() / "Library/Caches/com.mkmasker.pro/temp_frames"}
    CLICK_MEMORY = {"points": {}, "labels": {}}
    if masks_dir.exists(): shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    ai_ann, _ = _get_ai()
    if ai_ann: ai_ann.clear_cache()
    await sio.emit('video_loaded', {'total_frames': total_f, 'fps': fps, 'width': w, 'height': h, 'first_frame_b64': get_frame_base64(0)}, to=sid)

@sio.on('select_model')
async def handle_select_model(sid, data):
    global SELECTED_MODEL
    SELECTED_MODEL = data.get('model', 'sam2')

@sio.on('request_frame')
async def handle_request_frame(sid, data):
    await sio.emit('frame_update', {'frame': data['frame'], 'image_b64': get_frame_base64(data['frame'])}, to=sid)

@sio.on('add_click')
async def handle_add_click(sid, data):
    if SELECTED_MODEL == "rvm": return 
    global CLICK_MEMORY
    ai_ann, _ = _get_ai(); f_idx = data['frame']
    CLICK_MEMORY["points"].setdefault(f_idx, []).append([data['x'] * SESSION['scale'], data['y'] * SESSION['scale']])
    CLICK_MEMORY["labels"].setdefault(f_idx, []).append(1 if data['is_positive'] else 0)
    img_path = SESSION['frames_dir'] / f"{f_idx:08d}.jpg"
    ai_ann.set_image(np.array(Image.open(img_path).convert("RGB")), f_idx)
    mask_pil = ai_ann.predict(CLICK_MEMORY["points"][f_idx], CLICK_MEMORY["labels"][f_idx])
    mask_pil.save(masks_dir / f"mask_{f_idx:04d}.png")
    
    full_res_m = mask_pil.resize((SESSION['width'], SESSION['height']), Image.NEAREST); m_np = np.array(full_res_m)
    rgba = np.zeros((m_np.shape[0], m_np.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 2] = 255; rgba[:, :, 3] = (m_np > 0) * 150
    buf = io.BytesIO(); Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    await sio.emit('mask_preview', {'frame': f_idx, 'mask_alpha_b64': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"}, to=sid)

@sio.on('start_processing')
async def handle_start_processing(sid, data):
    global CLICK_MEMORY, SELECTED_MODEL
    _, ai_run = _get_ai(); main_loop = asyncio.get_running_loop(); mode = data.get('format', 'prores')

    def on_prog(f, t):
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
            ai_run.reset()
            for i, f_path in enumerate(frame_files):
                mask_np = ai_run.process_frame(cv2.imread(str(f_path)))
                Image.fromarray(mask_np, mode="L").save(masks_dir / f"mask_{i:04d}.png")
                if i % 5 == 0: on_prog(i, len(frame_files))
        else:
            if not CLICK_MEMORY["points"]: raise NoSelectionError(
                "No selections found. Click on the subject in the video before processing.")

            # ✅ FIX: Explicitly delete all UI preview masks so ExportPipeline doesn't hit weird artifacts
            for f in masks_dir.glob("mask_*.png"): f.unlink()

            # The runner will now cleanly output mask_0000.png directly
            await main_loop.run_in_executor(None, ai_run.run_bidirectional, frame_files, CLICK_MEMORY["points"], CLICK_MEMORY["labels"], masks_dir, on_prog)

        # Export Pipeline
        out_path = Path(data['output_dir']) / f"cutout_{Path(SESSION['path']).stem}"
        render_temp = Path.home() / "Library/Caches/com.mkmasker.pro/render_temp"
        if render_temp.exists(): shutil.rmtree(render_temp)
        render_temp.mkdir(parents=True)

        pipeline = ExportPipeline()
        label_hint = "Rendering alpha masks"
        pipeline.render_matte_frames(masks_dir, render_temp, on_progress=on_export_prog, on_phase=on_phase)
        if mode in ["prores", "balanced"]:
            label_hint = "Preparing RGBA frames"
            pipeline.prepare_rgba_frames(SESSION['frames_dir'], render_temp, render_temp, SESSION['total_frames'], on_progress=on_export_prog, on_phase=on_phase)
        label_hint = f"Encoding {mode}"
        final_file = pipeline.encode_video(render_temp, out_path, SESSION['fps'], mode, on_phase=on_phase)
        await sio.emit('process_complete', {'output_files': [str(final_file)]}, to=sid)
        await sio.emit('phase_update', {'phase': 'Done'}, to=sid)
        shutil.rmtree(render_temp)

    except Exception as e:
        err = as_mk_error(e)
        logging.error(f"Render Error [{err.code}]: {err.user_message}")
        if err.detail:
            logging.error(f"Detail: {err.detail}")
        await sio.emit('error_alert', err.to_payload(), to=sid)
        await sio.emit('phase_update', {'phase': 'Error'}, to=sid)

if __name__ == '__main__': web.run_app(app, port=8080)