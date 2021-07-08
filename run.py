from utils import *
from model import *

import torch
import os
import sys


def load_batch(filename):
    images = []
    files = []
    folder = os.path.dirname(filename)
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line[:-1]
            images.append(load_image(os.path.join(folder, line)))
            files.append(line)

    return torch.stack(images, dim=0), files


def run(batch_file):
    images, files = load_batch(batch_file)
    model = PokeCNN(4)
    model.load_state_dict(torch.load('output/model.pt'))

    device = torch.device("cuda")
    model.to(device)

    predictions = classify(model, images.to(device))

    for i, prediction in enumerate(predictions):
        print("File {} was class {}".format(files[i], prediction))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Missing files to classify")
        exit(0)

    run(sys.argv[1])
