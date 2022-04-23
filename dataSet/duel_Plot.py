from utils import *


def load_Npy(npy_path):
    train_acc = np.load(npy_path + "train_acc.npy", encoding="latin1")
    valid_acc = np.load(npy_path + "valid_acc.npy", encoding="latin1")

    train_loss = np.load(npy_path + "train_loss.npy", encoding="latin1")
    valid_loss = np.load(npy_path + "valid_loss.npy", encoding="latin1")

    return (train_acc, valid_acc), (train_loss, valid_loss)


def full_Plot():
    size = 18

    (train_acc1, valid_acc1), (train_loss1, valid_loss1) = load_Npy(r"./无数据增强MNIST/LeNet5/")
    train_loss1 = np.array(train_loss1)
    valid_loss1 = np.array(valid_loss1)
    train_acc1 = np.array(train_acc1)
    valid_acc1 = np.array(valid_acc1)

    (train_acc2, valid_acc2), (train_loss2, valid_loss2) = load_Npy(r"./无数据增强MNIST/ResNet18/")
    train_loss2 = np.array(train_loss2)
    valid_loss2 = np.array(valid_loss2)
    train_acc2 = np.array(train_acc2)
    valid_acc2 = np.array(valid_acc2)

    loss1_max = np.max(train_loss1) if np.max(train_loss1) > np.max(valid_loss1) else np.max(valid_loss1)
    loss2_max = np.max(train_loss2) if np.max(train_loss2) > np.max(valid_loss2) else np.max(valid_loss2)
    loss_max = loss1_max if loss1_max > loss2_max else loss2_max

    acc1_min = np.min(train_acc1) if np.min(train_acc1) < np.min(valid_acc1) else np.min(valid_acc1)
    acc2_min = np.min(train_acc2) if np.min(train_acc2) < np.min(valid_acc2) else np.min(valid_acc2)
    acc_min = acc1_min if acc1_min < acc2_min else acc2_min

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), tight_layout=True)  # 创建一个包含一个axes的figure

    ax[0].set(ylim=(0.00, loss_max))
    ax[1].set(ylim=(0.00, loss_max))

    l1 = ax[0].plot(train_loss1, '--', color='blue', label="Train loss")
    l2 = ax[0].plot(valid_loss1, '--', color='red', label="Valid loss")

    ax2 = ax[0].twinx()
    ax2.set(ylim=(acc_min, 1.00))

    l3 = ax2.plot(train_acc1, color='blue', label="Train acc")
    l4 = ax2.plot(valid_acc1, color='red', label="Valid acc")

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc='upper left')

    EPOCHS = np.array(valid_acc1).size
    if EPOCHS == 15:
        blink = 2
    elif EPOCHS == 50:
        blink = 7
    elif EPOCHS == 100:
        blink = 11
    elif EPOCHS == 240:
        blink = 34
    else:
        blink = 18
    ticks = list(range(0, EPOCHS, blink))
    if EPOCHS == 15 or EPOCHS == 50 or EPOCHS == 100 or EPOCHS == 200:
        pass
    else:
        ticks[-1] = EPOCHS

    tickl = [i + 1 for i in ticks]
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(tickl, rotation=10, fontsize=size - 5)
    plt.tick_params(labelsize=size - 5)

    ax[0].set_xlabel('Epoch', fontsize=size)
    ax[0].set_ylabel("Loss", fontsize=size)

    ax[0].grid(b=False, axis="y")
    ax2.grid(b=False, axis="y")
    ax[0].grid(b=True, axis="x")

    # subplot(2)

    l1 = ax[1].plot(train_loss2, '--', color='blue', label="Train loss")
    l2 = ax[1].plot(valid_loss2, '--', color='red', label="Valid loss")

    ax2 = ax[1].twinx()
    ax2.set(ylim=(acc_min, 1.00))

    l3 = ax2.plot(train_acc2, color='blue', label="Train acc")
    l4 = ax2.plot(valid_acc2, color='red', label="Valid acc")

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc='upper left')

    EPOCHS = np.array(valid_acc2).size
    if EPOCHS == 15:
        blink = 2
    elif EPOCHS == 50:
        blink = 7
    elif EPOCHS == 100:
        blink = 11
    elif EPOCHS == 240:
        blink = 34
    else:
        blink = 18
    ticks = list(range(0, EPOCHS, blink))
    if EPOCHS == 15 or EPOCHS == 50 or EPOCHS == 100:
        pass
    else:
        ticks[-1] = EPOCHS - 1

    print(ticks)

    tickl = [i + 1 for i in ticks]
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(tickl, rotation=10, fontsize=size - 5)
    plt.tick_params(labelsize=size - 5)

    ax[1].set_xlabel('Epoch', fontsize=size)

    ax2.set_ylabel("Acc", fontsize=size)

    ax[1].grid(b=False, axis="y")
    ax2.grid(b=False, axis="y")
    ax[1].grid(b=True, axis="x")

    Path = r"./无数据增强MNIST"
    fig.savefig(Path + "/" + "无数据增强.png", bbox_inches='tight', pad_inches=0)
    fig.show()
    plt.style.use('default')


full_Plot()
