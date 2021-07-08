import torch
import cv2

IM_H = 128
IM_W = 128


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IM_H, IM_W))
    return torch.tensor(img.transpose(2, 0, 1) / 255.0).float()


def test_accuracy(model, x, y):
    predictions = classify(model, x)
    correct = predictions.eq(y).sum().item()
    return correct


def classify(model, x):
    output = model(x)
    return torch.max(output, 1).indices
