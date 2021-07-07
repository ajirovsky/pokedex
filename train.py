from dataset import *
from utils import *
from model import PokeCNN

import math
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

classes = ['Pikachu', 'Charmander', 'Bulbasaur', 'Squirtle']
dataset = PokeDataset('dataset/', classes)

train_ratio = 0.9
n_epochs = 20
batch_size = 32

train_size = math.floor(train_ratio * len(dataset))
test_size = math.ceil((1 - train_ratio) * len(dataset))

print("Training set of {} images and testing set of {} images".format(train_size, test_size))

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda")
model = PokeCNN(class_count=len(classes)).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

train_accs = []
test_accs = []

early_stopper = {'best_acc': 0, 'counter': 0, 'max_counter': 3}

for i in range(n_epochs):
    model.train()
    for i_batch, (img, label) in enumerate(train_loader):
        x, y = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if i_batch % 50 == 0:
            print('Epoch: {}, batch: {}, loss: {}'.format(i, i_batch, loss.item()))

    model.eval()

    with torch.no_grad():
        batch_acc = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_acc += test_accuracy(model, x, y)
        train_accs.append(batch_acc / len(train_loader.dataset))

        batch_acc = 0

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_acc += test_accuracy(model, x, y)
        test_accs.append(batch_acc / len(test_loader.dataset))

        if batch_acc > early_stopper['best_acc']:
            early_stopper['best_acc'] = batch_acc
            early_stopper['counter'] = 0

            if not os.path.exists('output'):
                os.mkdir('output')
            torch.save(model.state_dict(), os.path.join('output', 'model.pt'))
        else:
            early_stopper['counter'] += 1

        if early_stopper['counter'] > early_stopper['max_counter']:
            print('Early stopper at {:d} epoch with accuracy {:.2f}'.format(i, early_stopper['best_acc']))
            break


plt.plot(range(len(train_accs)), train_accs)
plt.plot(range(len(test_accs)), test_accs)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train', 'test'])
plt.show()
