import os
import numpy as np
import matplotlib.pyplot as plt


def plot_es(N_EPOCHS, train, valid, Style, unix_timestamp, savePath='./png/'):
    plt.style.use('seaborn')

    train = np.array(train)
    valid = np.array(valid)

    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)  # 创建一个包含一个axes的figure

    if Style == "loss":
        a1label = 'Training loss'
        a2label = 'Validation loss'
        title = 'Loss over epochs'
        ylabel = 'Loss'
    if Style == "acc":
        a1label = 'Training Acc'
        a2label = 'Validation Acc'
        title = 'Acc over epochs'
        ylabel = 'Acc'

    ax.plot(train, color='blue', label=a1label)
    ax.plot(valid, color='red', label=a2label)
    ax.set(title=title,
           xlabel='Epoch',
           ylabel=ylabel)

    if N_EPOCHS == 50:
        blink = 7
    else:
        blink = 3

    ticks = list(range(0, N_EPOCHS, blink))
    ticks.append(N_EPOCHS - 1)
    tickla = [i + 1 for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tickla, rotation=10)
    ax.legend()

    unix_timestamp = str(int(unix_timestamp))
    Path = savePath + unix_timestamp
    isExists = os.path.exists(Path)
    if not isExists:
        os.makedirs(Path)

    fig.savefig(Path + "/" + Style + '.png')
    np.save(Path + "/" + Style + '.npy', train)
    np.save(Path + "/" + Style + '.npy', valid)

    fig.show()
    plt.style.use('default')


def full_plot(N_EPOCHS, train_loss, valid_loss, train_acc, valid_acc, unix_timestamp, savePath='./png/'):
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    train_acc = np.array(train_acc)
    valid_acc = np.array(valid_acc)

    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)  # 创建一个包含一个axes的figure

    l1 = ax.plot(train_loss, '--', color='blue', label="Train loss")
    l2 = ax.plot(valid_loss, '--', color='red', label="Valid loss")

    ax2 = ax.twinx()

    l3 = ax2.plot(train_acc, color='blue', label="Train acc")
    l4 = ax2.plot(valid_acc, color='red', label="Valid acc")

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')

    if N_EPOCHS == 50:
        blink = 7
    else:
        blink = 2
    ticks = list(range(0, N_EPOCHS, blink))
    ticks.append(N_EPOCHS - 1)
    tickl = [i + 1 for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickl, rotation=10)

    ax.set(title="Acc & Loss over Epochs", xlabel='Epoch', ylabel="Loss")

    ax2.set(ylabel="Acc")

    ax.grid(b=False, axis="y")
    ax2.grid(b=False, axis="y")
    ax.grid(b=True, axis="x")

    Path = savePath + str(unix_timestamp)
    if not os.path.exists(Path):
        os.makedirs(Path)
    fig.savefig(Path + "/" + "fig.png")
    fig.show()
    plt.style.use('default')


def loadNpy(unix_timestamp, Path="./png/"):
    npy_path = Path + str(unix_timestamp) + "/"

    train_acc = np.load(npy_path + "train_acc.npy", encoding="latin1")
    valid_acc = np.load(npy_path + "valid_acc.npy", encoding="latin1")

    train_loss = np.load(npy_path + "train_loss.npy", encoding="latin1")
    valid_loss = np.load(npy_path + "valid_loss.npy", encoding="latin1")

    return (train_acc, valid_acc), (train_loss, valid_loss)


def saveNpy(unix_timestamp, train_loss, valid_loss, train_acc, valid_acc, Path="./png/"):
    npy_path = Path + str(unix_timestamp) + "/"

    np.save(npy_path + "train_acc" + ".npy", train_acc)
    np.save(npy_path + "valid_acc" + ".npy", valid_acc)
    np.save(npy_path + "train_loss" + ".npy", train_loss)
    np.save(npy_path + "valid_loss" + ".npy", valid_loss)
