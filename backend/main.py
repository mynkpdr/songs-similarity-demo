import os
import shutil
import tempfile
import subprocess
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp

app = FastAPI(title="SonicLens Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/download")
def download_video(url: str = Form(...)):
    try:
        # TEMPORARY: yt-dlp is not working, use local samples
        url_mapping = {
            "https://www.youtube.com/watch?v=GX9x62kFsVU": ("gehra_hua.mp3", "Gehra Hua | Dhurandhar"),
            "https://www.youtube.com/watch?v=Ans8Y59cvds": ("phir_se.mp3", "PHIR SE | Dhurandhar The Revenge"),
            "https://www.youtube.com/watch?v=nDjloeIB3Pc": ("sitaare.mp3", "Sitaare | Ikkis"),
            "https://www.youtube.com/watch?v=ko70cExuzZM": ("the_fate_of_ophelia.mp3", "The Fate of Ophelia"),
            "https://www.youtube.com/watch?v=1FVF-9KQiPo": ("opalite.mp3", "Opalite"),
        }

        if url in url_mapping:
            filename, title = url_mapping[url]
            mp3_path = os.path.join(os.path.dirname(__file__), "samples", filename)
        else:
            # Fallback to default sample
            mp3_path = os.path.join(os.path.dirname(__file__), "audio.mp3")
            title = "Woh Din Bhi Kya Din The"

        if not os.path.exists(mp3_path):
            # Try to find any sample as fallback
            samples_dir = os.path.join(os.path.dirname(__file__), "samples")
            if os.path.exists(samples_dir) and os.listdir(samples_dir):
                first_sample = os.listdir(samples_dir)[0]
                mp3_path = os.path.join(samples_dir, first_sample)
                title = "Sample Audio"
            else:
                raise HTTPException(status_code=500, detail="Sample audio file not found")
        
        return FileResponse(
            mp3_path, 
            media_type='audio/mpeg', 
            filename="audio.mp3",
            headers={"Access-Control-Expose-Headers": "X-Audio-Title", "X-Audio-Title": title.encode('ascii', 'ignore').decode()}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/cut")
def cut_audio(file: UploadFile = File(...), start_time: str = Form(...), end_time: str = Form(...)):
    try:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input.mp3")
        output_path = os.path.join(temp_dir, "output.mp3")
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Parse times to check duration
        def parse_time(t_str):
            parts = t_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
            
        s_time = parse_time(start_time)
        e_time = parse_time(end_time)
        if e_time - s_time > 60:
            raise HTTPException(status_code=400, detail="Cut duration cannot exceed 1 minute.")
            
        cmd = [
            "ffmpeg", "-y", "-i", input_path, 
            "-ss", start_time, "-to", end_time, 
            "-c", "copy", output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if not os.path.exists(output_path):
             raise HTTPException(status_code=500, detail="FFmpeg failed to cut the audio.")
             
        return FileResponse(output_path, media_type='audio/mpeg', filename="cut_audio.mp3")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/separate")
def separate_audio(file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input.mp3")
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        output_dir = os.path.join(temp_dir, "separated")
        
        cmd = [
            "demucs", "-n", "htdemucs_6s", "-o", output_dir, input_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        stem_dir = os.path.join(output_dir, "htdemucs_6s", "input")
        
        if not os.path.exists(stem_dir):
            raise HTTPException(status_code=500, detail="Demucs failed to generate stems.")
            
        stems = {}
        stems_low = {}
        
        for stem_file in os.listdir(stem_dir):
            if stem_file.endswith(".wav"):
                stem_name = Path(stem_file).stem
                wav_path = os.path.join(stem_dir, stem_file)
                
                # 1. High Quality for Embeddings (MP3 192k or high-quality OGG)
                hq_path = os.path.join(stem_dir, f"{stem_name}_hq.mp3")
                subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", hq_path], check=True, capture_output=True)
                with open(hq_path, "rb") as f:
                    stems[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                
                # 2. Low Quality for Storage (Opus 48k)
                lq_path = os.path.join(stem_dir, f"{stem_name}_lq.ogg")
                subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "48k", lq_path], check=True, capture_output=True)
                with open(lq_path, "rb") as f:
                    stems_low[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                    
        # Original cut audio
        full_hq_path = os.path.join(temp_dir, "full_hq.mp3")
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-b:a", "192k", full_hq_path], check=True, capture_output=True)
        with open(full_hq_path, "rb") as f:
            stems['full'] = base64.b64encode(f.read()).decode('utf-8')
            
        full_lq_path = os.path.join(temp_dir, "full_lq.ogg")
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-c:a", "libopus", "-b:a", "32k", full_lq_path], check=True, capture_output=True)
        with open(full_lq_path, "rb") as f:
            stems_low['full'] = base64.b64encode(f.read()).decode('utf-8')
            
        return JSONResponse(content={"stems": stems, "stems_low": stems_low})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/embeddings/{artist}/{stem}")
def get_embedding(artist: str, stem: str):
    # Look for the file in the audios directory
    # Depending on where this is deployed, we look one or two levels up
    for base_dir in [
        os.path.join(os.path.dirname(__file__), '..', 'audios'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'audios'),
        os.path.join(os.path.dirname(__file__), 'audios'),
    ]:
        path = os.path.join(base_dir, artist, 'embeddings', f"{stem}.npy")
        if os.path.exists(path):
            return FileResponse(path)
            
    raise HTTPException(status_code=404, detail="Embedding not found")
