import gradio as gr
import cv2
import tempfile
import subprocess
from pathlib import Path
from client import ShredMLClient
import os
from random import shuffle
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

client = ShredMLClient()

# video wont play without this
def convert_video(input_path):
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temp_output.name
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return temp_output.name


def plot_probabilities(probs):
    if not probs:
        return None

    classes = list(probs.keys())
    values = [probs[c] for c in classes]

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(classes, values, color=['skyblue', 'orange'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


print(os.listdir())

DATA_DIR = "../data"
TRICK_CLASSES = os.listdir(DATA_DIR)


EXAMPLES = []

for trick_class in TRICK_CLASSES:
    trick_videos_dir = os.path.join(DATA_DIR, trick_class)
    for video in os.listdir(trick_videos_dir):
        EXAMPLES.append(os.path.join(trick_videos_dir, video)) 

shuffle(EXAMPLES)


def run_inference(video_path):
    job_id = client.upload_video(video_path)
    result = client.wait_for_result(job_id)

    probs = result.get("probabilities", {})
    top_class = max(probs, key=probs.get) if probs else "Unknown"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    bboxes = result.get("bboxes", [])
    bbox_by_frame = {}
    for b in bboxes:
        bbox_by_frame.setdefault(b["frame"], []).append(b)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for b in bbox_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = map(int, b["bbox"])
            conf = b.get("confidence", 1.0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{top_class} ({conf:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return convert_video(temp_output.name)


with gr.Blocks() as demo:
    gr.Markdown("# ShredML")

    with gr.Row():
        with gr.Column(scale=1, min_width=150):
            video_input = gr.File(label="Upload your video", file_types=[".mp4", ".mov"])

            example_dropdown = gr.Dropdown(
                label="Or select an example video",
                choices=[Path(v).name for v in EXAMPLES],
                value=None
            )

            run_btn = gr.Button("Run Inference", elem_classes="fullwidth")
            
            prob_plot = gr.Image(label="Class Probabilities")

        with gr.Column(scale=2, min_width=300):
            output_video = gr.Video(label="Result with Bounding Boxes", height=720)

    def on_upload_change(file):
        example_dropdown.value = None

    video_input.change(on_upload_change, inputs=video_input, outputs=example_dropdown)

    def process(video_file, selected_example_name):
        if selected_example_name:
            path = next(v for v in EXAMPLES if Path(v).name == selected_example_name)
        elif video_file:
            path = video_file.name
        else:
            return None, None

        result_video = run_inference(path)

        job_id = client.upload_video(path)
        result = client.wait_for_result(job_id)
        prob_image = plot_probabilities(result.get("probabilities", {}))

        return result_video, prob_image

    run_btn.click(process, inputs=[video_input, example_dropdown], outputs=[output_video, prob_plot])

demo.launch(server_port=7860)
