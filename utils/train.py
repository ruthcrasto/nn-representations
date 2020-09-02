import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.cka import feature_space_linear_cka


def validate(model, data_loader):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    acc = 0

    for i, (x, t) in enumerate(data_loader):

        pred = model(x)
        loss = loss_fn(pred, t)
        pred = np.argmax(pred.detach().numpy(), axis=1)
        targ = np.array(t)
        total_loss += loss.item()
        acc += np.sum(pred == targ)

    return total_loss / len(data_loader), acc / len(data_loader.dataset)


def train(model, train_loader, valid_loader, hp):

    print("Beginning Training...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    total = 0
    train_losses = []
    valid_losses = []
    representations = []
    num_iters = hp["num_iters"]


    # Training loop
    for itr in range(num_iters):

        for i, (x, t) in enumerate(train_loader):

            optimizer.zero_grad()

            if len(representations) == itr:
                representations.append(model.get_representation(x))

            y = model(x)
            loss = loss_fn(y, t)
            total += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total / len(train_loader)
        print("average training loss {}: ".format(itr), avg_loss)
        train_losses.append(avg_loss)


        valid_loss, acc = validate(model, valid_loader)
        print("validation accuracy {}: ".format(itr), round(acc * 100, 3), "%")
        valid_losses.append(valid_loss)

        total = 0

    # Save representation similarities
    n = len(representations)
    similarities = np.zeros((n, n))

    for i in range(n):
        repr1 = representations[i].detach().numpy().flatten()

        for j in range(i + 1, n):
            repr2 = representations[j].detach().numpy().flatten()
            similarities[i, j] = feature_space_linear_cka(repr1, repr2)

    similarities = similarities + similarities.T + np.eye(n)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(np.arange(num_iters) + 1, train_losses, label="Training loss")
    ax1.plot(np.arange(num_iters) + 1, valid_losses, label="Validation loss")
    ax2.plot(np.arange(num_iters) + 1, similarities[n - 1], label="Similarity", color="green")
    ax1.legend()
    ax2.legend()

    plt.show()
