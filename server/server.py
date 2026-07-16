import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path: sys.path.insert(0, str(current_dir))

import socketio, asyncio, os, shutil, numpy as np, base64, io, logging, re
from aiohttp import web
from PIL import Image
from core.sam2_runner import SAM2ImageAnnotator, SAM2VideoRunner
from utils.video_handler import extract_frames, get_frame_base64
from utils.export import ExportPipeline

logging.basicConfig(level=logging.INFO)
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application(); sio.attach(app)

# --- GLOBAL STATE ---
SESSION = {}
CLICK_MEMORY = {"points": {}, "labels": {}} 
masks_dir = Path.home() / "Library/Caches/com.mkmasker.pro/masks"
annotator = None
video_runner = None

def _get_ai():
    global annotator, video_runner
    if annotator is None:
        annotator = SAM2ImageAnnotator(variant="small")
        video_runner = SAM2VideoRunner(variant="small")
    return annotator, video_runner

@sio.event
async def connect(sid, environ):
    logging.info(f"🟢 Connected: {sid}")

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
    ai_ann, _ = _get_ai(); ai_ann.clear_cache()
    await sio.emit('video_loaded', {'total_frames': total_f, 'fps': fps, 'width': w, 'height': h, 'first_frame_b64': get_frame_base64(0)}, to=sid)
    await sio.emit('system_status', {'status': 'Ready'}, to=sid)

@sio.on('request_frame')
async def handle_request_frame(sid, data):
    f_idx = data.get('frame', 0)
    await sio.emit('frame_update', {'frame': f_idx, 'image_b64': get_frame_base64(f_idx)}, to=sid)

@sio.on('add_click')
async def handle_add_click(sid, data):
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

# ... (imports and global state stay exactly same as last stable) ...

@sio.on('start_processing')
async def handle_start_processing(sid, data):
    global CLICK_MEMORY
    _, runner = _get_ai(); main_loop = asyncio.get_running_loop()
    
    # UI selected format (prores, balanced, bw)
    mode = data.get('format', 'prores')
    
    def on_prog(f, t):
        p = int((f/SESSION['total_frames'])*100)
        asyncio.run_coroutine_threadsafe(sio.emit('progress_update', {'percentage': p, 'message': f"Tracking... {f}/{SESSION['total_frames']}"}, to=sid), main_loop)

    try:
        if not CLICK_MEMORY["points"]: raise ValueError("No selections found.")
        seed_frame = sorted(CLICK_MEMORY["points"].keys())[0]
        points, labels = CLICK_MEMORY["points"][seed_frame], CLICK_MEMORY["labels"][seed_frame]
        frame_files = sorted(list(SESSION['frames_dir'].glob("*.jpg")))
        
        # 1. AI Tracking
        await main_loop.run_in_executor(None, runner.run_bidirectional, frame_files, seed_frame, points, labels, masks_dir, on_prog)

        # 2. Workspace Setup
        out_path = Path(data['output_dir']) / f"cutout_{Path(SESSION['path']).stem}"
        render_temp = Path.home() / "Library/Caches/com.mkmasker.pro/render_temp"
        if render_temp.exists(): shutil.rmtree(render_temp)
        render_temp.mkdir(parents=True)
        
        pipeline = ExportPipeline()
        
        # 3. Step A: Create clean matte sequence
        await sio.emit('progress_update', {'percentage': 92, 'message': "Cleaning Matte Frames..."}, to=sid)
        pipeline.render_matte_frames(masks_dir, render_temp)
        
        # 4. Step B: Handle Alpha Merging if needed
        if mode in ["prores", "balanced"]:
            await sio.emit('progress_update', {'percentage': 95, 'message': "Merging Alpha Channel..."}, to=sid)
            pipeline.prepare_rgba_frames(SESSION['frames_dir'], render_temp, render_temp, SESSION['total_frames'])
        
        # 5. Final Encode
        await sio.emit('progress_update', {'percentage': 98, 'message': f"Encoding {mode.upper()}..."}, to=sid)
        final_file = pipeline.encode_video(render_temp, out_path, SESSION['fps'], mode)
        
        await sio.emit('process_complete', {'output_files': [str(final_file)]}, to=sid)
        shutil.rmtree(render_temp)
        
    except Exception as e:
        logging.error(e); await sio.emit('error_alert', {'message': str(e)}, to=sid)

# ... (main entry point same as last stable) ...

if __name__ == '__main__': web.run_app(app, port=8080)