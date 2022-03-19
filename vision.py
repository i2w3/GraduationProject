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
    tickla = [i+ 1 for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tickla, rotation=10)
    ax.legend()

    unix_timestamp = str(int(unix_timestamp))
    print("当前时间戳为", unix_timestamp)
    Path = savePath + unix_timestamp
    isExists = os.path.exists(Path)
    if not isExists:
        os.makedirs(Path)

    fig.savefig(Path + "/" + Style + '.png')
    np.save(Path + "/" + Style + '.npy', train)
    np.save(Path + "/" + Style + '.npy', valid)

    fig.show()
    plt.style.use('default')
