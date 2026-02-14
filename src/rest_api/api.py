from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import shutil
from uuid import uuid4
import os
from typing import Dict

from ml.predict import TrickEstimator

app = FastAPI()

VIDEO_DIR = Path("tmp_estimator")
VIDEO_DIR.mkdir(exist_ok=True)

classifier = TrickEstimator()

jobs: Dict[str, dict] = {}


def run_inference(job_id: str, video_path: Path):
    try:
        jobs[job_id]["status"] = "processing"

        result = classifier.predict(str(video_path))

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

    finally:
        # cleanup
        try:
            video_path.unlink(missing_ok=True)
            video_path.parent.rmdir()
        except Exception:
            pass


@app.post("/inference/")
async def do_inference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    job_id = str(uuid4())

    req_dir = VIDEO_DIR / job_id
    req_dir.mkdir(exist_ok=True)

    print(file.filename)
    print(hash(file))

    path = Path(file.filename).stem

    save_path = req_dir / path

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {
        "status": "queued",
        "result": None,
    }

    background_tasks.add_task(run_inference, job_id, save_path)

    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.get("/inference/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    os.popen(f"rm -rf {VIDEO_DIR}/{job_id}")

    return jobs[job_id]
