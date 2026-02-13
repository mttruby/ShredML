import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import KFold

from ml.nodes.feature.feature_extractor import FeatureExtractor
from ml.nodes.classifier.lstm_classifier import LSTMClassifier
from ml.train.load_data import get_data_loader

def train_lstm_cv(k=5, num_epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    full_loader = get_data_loader()
    dataset = list(full_loader.dataset)  
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []

    fe = FeatureExtractor()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{k}")
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=16, shuffle=False)

        model = LSTMClassifier(feature_extractor=fe, num_classes=2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        metrics = {"loss": [], "accuracy": []}

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for frames, labels in train_loader:
                if frames is None or labels is None:
                    continue
                frames = frames.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                preds = model(frames)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * frames.size(0)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            metrics["loss"].append(epoch_loss)
            metrics["accuracy"].append(epoch_acc)

            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f}")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                preds = model(frames)
                _, predicted = torch.max(preds, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"Validation Accuracy Fold {fold+1}: {val_acc:.4f}")

        all_metrics.append({"train": metrics, "val_acc": val_acc})

        torch.save(model.state_dict(), f"lstm_model_fold{fold+1}.pt")

    with open("metrics_cv.pkl", "wb") as f:
        pickle.dump(all_metrics, f)



if __name__ == "__main__":
    train_lstm_cv()
