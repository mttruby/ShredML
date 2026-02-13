from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
import cv2
import numpy as np
import os
import torch 
from PIL import Image


# TODO: find out why length of frames was 0 for at least 1 sample
# TODO: find out why Ollie83 and Ollie91 do not contain images after ppr

SEQUENCE_LENGTH = 40

class TrickDataset(Dataset):

    def __init__(self, root_dir, label_map):
        self.samples = []
        self.label_map = label_map

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)

            for video in os.listdir(label_dir):
                video_dir = os.path.join(label_dir, video)

                images = sorted(glob.glob(video_dir + "/*.jpg"))

                self.samples.append((images, label))

                print(f"{video_dir}: {label}")

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        image_paths, label = self.samples[idx]

        step = max(1, len(image_paths) // SEQUENCE_LENGTH)
        selected = image_paths[::step][:SEQUENCE_LENGTH]

        frames = []
        for p in selected:
            img = cv2.imread(p)
            if img is None:
                continue  # skip if 0 frames/no images in dir or whatever is causing it
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            frames.append(img)

        if len(frames) == 0:
            return None  # handle it in dataloader
        
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])

        frames = torch.stack(frames)
        label = self.label_map[label]

        return frames, label
