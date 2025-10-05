import torch
from model.lightweight_model import LightweightCNN
from utils.dataloader import load_data
from utils.metrics import evaluate_model

def main():
    model = LightweightCNN()
    train_loader, val_loader, test_loader = load_data(batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    # Evaluation
    model.eval()
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
