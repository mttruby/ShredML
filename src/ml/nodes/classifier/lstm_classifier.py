# based on the architecture from LightningDrop/SkateboardML
# see: https://github.com/LightningDrop/SkateboardML

import torch.nn as nn

from ml.nodes.feature.feature_extractor import FeatureExtractor

class LSTMClassifier(nn.Module):

    def __init__(self, feature_extractor: FeatureExtractor, num_classes):
        super().__init__()

        self.feature_extractor = feature_extractor

        self.lstm = nn.LSTM(
            input_size=feature_extractor.out_dim,
            hidden_size=256,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        last = lstm_out[:, -1]
        return self.classifier(last)
