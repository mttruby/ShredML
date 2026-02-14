import cv2
import json
import os
from ultralytics import YOLO
from typing import Union
from pathlib import Path
import torch
import logging
from tqdm import tqdm


class YOLOPreprocessor:
    """
    Wraps YOLO model(s) to use it for bounding box estimation, this will the be used to crop the image down to the bbox. 
    """

    def __init__(
        self,
        yolo_model: str = "yolo26n.pt",
        confidence_threshold: float = 0.5,
        cropped_images_dir: str = "outputs/cropped_images",
        bbox_dir: str = "outputs/bounding_boxes",
        use_gpu: bool = False,
    ):

        self.model = YOLO(
            yolo_model,
        )
        self.conf_thresh = confidence_threshold
        self.cropped_dir = cropped_images_dir
        self.bbox_dir = bbox_dir

        self._skateboard_class = [k for k, v in self.model.names.items() if v == "skateboard"][0]
        self._person_class = [k for k, v in self.model.names.items() if v == "person"][0]

        if use_gpu and torch.backends.cudnn.is_available():
            self.model.to("cuda")


    def load_video(self, filepath: Union[str, Path]):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise FileNotFoundError(f"File not found: {filepath}")
        

    def get_bounding_boxes_and_cropped_images(
        self, cap: cv2.VideoCapture, video_name: str = None, trick_label: str = None, output_dir: str = None,

    ):

        frame_id = 0
        bounding_boxes = []
        cropped_images = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.track(frame, persist=True, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls != self._skateboard_class or conf < self.conf_thresh:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    track_id = int(box.id[0]) if box.id is not None else -1
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    bounding_boxes.append(
                        {
                            "frame": frame_id,
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "center": [cx, cy],
                            "confidence": conf,
                        }
                    )

                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        cropped_images.append(crop)
                        
                        if output_dir is None:
                            
                            _cropped_dir = f"{self.cropped_dir}/{trick_label}/{video_name}"
                            
                            os.makedirs(_cropped_dir, exist_ok=True)
                            os.makedirs(self.bbox_dir, exist_ok=True)

                            cv2.imwrite(
                                f"{_cropped_dir}/_frame{frame_id}_track{track_id}.jpg", crop
                            )
                        else:
                            _cropped_dir = f"{output_dir}/frames"
                            os.makedirs(_cropped_dir, exist_ok=True)
                            cv2.imwrite(
                                f"{_cropped_dir}/_frame{frame_id}_track{track_id}.jpg", crop
                            )
            frame_id += 1

        cap.release()

        if output_dir is None:
            with open(f"{self.bbox_dir}/{video_name}.json", "w") as outf:
                json.dump(bounding_boxes, outf, indent=2)
        else:
            os.makedirs(f"{output_dir}/bounding_boxes", exist_ok=True)
            with open(f"{output_dir}/bounding_boxes/{video_name}.json", "w") as outf:
                json.dump(bounding_boxes, outf, indent=2)

        return bounding_boxes, cropped_images

    def preprocess_videos(self, video_input_dir: Union[str, Path]):
        print(f"Preprocessing videos with label: {Path(video_input_dir).stem}")
        for f in tqdm(os.listdir(video_input_dir)):
            filename = Path(f)
            filepath = f"{video_input_dir}/{filename}"
            try:
                self.get_bounding_boxes_and_cropped_images(
                    cap=cv2.VideoCapture(filepath),
                    video_name=filename.stem,
                    trick_label=Path(video_input_dir).stem,
                )
            except Exception as e:
                print(e)


if __name__ == "__main__":
    vp = YOLOPreprocessor(yolo_model="yolo26n.pt", use_gpu=True)
    vp.preprocess_videos("data/Video_Tricks/Kickflip")
    vp.preprocess_videos("data/Video_Tricks/Ollie")
