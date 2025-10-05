from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch

def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Accuracy: {acc*100:.2f}% | F1-score: {f1:.2f}")
