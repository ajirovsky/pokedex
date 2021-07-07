import torch


def test_accuracy(model, x, y):
    predictions = classify(model, x)
    correct = predictions.eq(y).sum().item()
    return correct


def classify(model, x):
    output = model(x)
    return torch.max(output, 1).indices
