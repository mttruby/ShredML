
from ml.nodes.preprocessor.yolo_preprocessor import YOLOPreprocessor
from ml.nodes.feature.feature_extractor import FeatureExtractor
from ml.nodes.classifier.lstm_classifier import LSTMClassifier
from typing import Union
from pathlib import Path
import os
from uuid import uuid4
import cv2
from ml.train.load_data import get_inference_loader
import torch
import torch.nn.functional as F


TMP_DIR="tmp_estimator"

class TrickEstimator:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.preprocessor = YOLOPreprocessor()
        self.feature_extractor = FeatureExtractor()

        self.model = LSTMClassifier(
            feature_extractor=self.feature_extractor,
            num_classes=2
        ).to(self.device)

        self.model.load_state_dict(
            torch.load("ml/models/small_model.pt", map_location=self.device)
        )

        self.model.eval()

        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)


    def predict(self, path_to_video: Union[str, Path]):

        cap = cv2.VideoCapture(str(path_to_video))
        parent_dir = Path(path_to_video).parent
        frames_dir = f"{parent_dir}/frames"
        os.makedirs(frames_dir)

        bboxes, _ = self.preprocessor.get_bounding_boxes_and_cropped_images(cap=cap, output_dir=parent_dir)

        loader = get_inference_loader(root_dir=f"{parent_dir}/frames")


        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue  # skip empty batches
                frames = batch
                frames = frames.to(self.device)
                logits = self.model(frames)
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                
                probs = F.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

            inv_label_map = {0:"Ollie", 1:"Kickflip"}

            result_dict = {}
            result_dict["probabilities"] = {}
            result_dict["bboxes"] = []
            result_dict["bboxes"] = bboxes

            for probs in probabilities:
                for i, p in enumerate(probs):
                    result_dict["probabilities"][inv_label_map[i]] = float(p) # numpy.float32' object is not iterable" 
            return result_dict


if __name__ == "__main__":
    pass
