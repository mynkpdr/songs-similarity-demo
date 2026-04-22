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

@app.get("/")
def read_root():
    return {"status": "online", "message": "SonicLens Processing Server is running"}

@app.post("/api/download")
def download_video(url: str = Form(...)):
    try:
        temp_dir = tempfile.mkdtemp()
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'YouTube Audio')
            
        # Find the resulting mp3 file
        mp3_path = None
        for f in os.listdir(temp_dir):
            if f.endswith(".mp3"):
                mp3_path = os.path.join(temp_dir, f)
                break
        
        if not mp3_path or not os.path.exists(mp3_path):
             raise HTTPException(status_code=500, detail="Failed to download or convert YouTube audio.")
             
        return FileResponse(
            mp3_path, 
            media_type='audio/mpeg', 
            filename="audio.mp3",
            headers={"Access-Control-Expose-Headers": "X-Audio-Title", "X-Audio-Title": title.encode('ascii', 'ignore').decode()}
        )
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
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Demucs Error: {result.stderr}")
        
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
