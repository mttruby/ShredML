import requests
import time
import mimetypes
from pathlib import Path

API_URL = "http://127.0.0.1"
API_PORT = 8500

class ShredMLClient:

    def __init__(self, api_url: str = API_URL, api_port: int = API_PORT):
        self.api_url = api_url
        self.api_port = api_port

    def upload_video(self, video_path: str) -> str:
        filename = Path(video_path).name 
        mime_type, _ = mimetypes.guess_type(video_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(video_path, "rb") as f:
            files = {
                "file": (filename, f, mime_type)
            }
            r = requests.post(f"{self.api_url}:{self.api_port}/inference/", files=files)

        r.raise_for_status()
        data = r.json()

        return data["job_id"]


    def wait_for_result(self, job_id: str, poll_interval=2):
        while True:
            r = requests.get(f"{self.api_url}:{self.api_port}/inference/{job_id}")
            r.raise_for_status()

            data = r.json()
            status = data["status"]

            print("Status:", status)

            if status == "done":
                return data["result"]

            if status == "failed":
                raise RuntimeError(data.get("error", "Inference failed"))

            time.sleep(poll_interval)


if __name__ == "__main__":
    video_path = "../data/Ollie/Ollie13.mov"

    client = ShredMLClient()

    print("Uploading video...")
    job_id = client.upload_video(video_path)
    print("Job ID:", job_id)

    print("Waiting for inference...")
    result = client.wait_for_result(job_id)

    print("\nResult:")
    print(result)
