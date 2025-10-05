import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = datasets.FakeData(transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size),
        DataLoader(test_data, batch_size=batch_size)
    )
