
from ml.nodes.preprocessor.yolo_preprocessor import YOLOPreprocessor
from ml.nodes.feature.feature_extractor import FeatureExtractor
from ml.nodes.classifier.lstm_classifier import LSTMClassifier
from typing import Union
from pathlib import Path
import os
from uuid import uuid4
import cv2
from ml.train.load_data import get_data_loader
import torch


TMP_DIR="tmp_estimator"

class TrickEstimator:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.preprocessor = YOLOPreprocessor()
        self.feature_extractor = FeatureExtractor()

        self.model = LSTMClassifier(
            feature_extractor=self.feature_extractor,
            num_classes=3
        ).to(self.device)

        self.model.load_state_dict(
            torch.load("ml/models/lstm_model_fold5.pt", map_location=self.device)
        )

        self.model.eval()

        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)


    def predict(self, path_to_video: Union[str, Path]):

        cap = cv2.VideoCapture(str(path_to_video))
        frame_dir = f"{Path(path_to_video).stem}/frames/cropped/"
        os.makedirs(frame_dir, exist_ok=False)

        bboxes, cimgs = self.preprocessor.get_bounding_boxes_and_cropped_images(cap=cap)

        for i, img in enumerate(cimgs):
            cv2.imwrite(f"{frame_dir}/frame_{i}.jpg", img)

        loader = get_data_loader(for_dir=Path(path_to_video).stem)
        print(f"Number of batches: {len(loader)}")


        predictions = []
        preds = None

        with torch.no_grad():
            for frames, _ in loader:

                frames = frames.to(self.device)
                logits = self.model(frames)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions, preds

        



# OUTPUT_CROPPED_IMGS = "outputs/cropped_images/"
# OUTPUT_BOUNDING_BOXES = "outputs/bounding_boxes/"

if __name__ == "__main__":
    
    # # Pipeline consisting of
    # # Preprocessing: YOLO for detecting the skateboard and cropping the image
    # # Feature extraction: a CNN (ResNet18) for feature extraction
    # # Classification: LSTM based sequential model
    

    # vp = YOLOPreprocessor(yolo_model="yolo26n.pt", use_gpu=True)
    # vp.preprocess_videos("data/Video_Tricks/Kickflip")
    # vp.preprocess_videos("data/Video_Tricks/Ollie")

    # cnn = FeatureExtractor()
    # classifier = LSTMClassifier(feature_extractor=cnn)

    pass
