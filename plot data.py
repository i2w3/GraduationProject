import os
from utils import *

dataPath = r"./png"

dataFolder = os.listdir(dataPath)

for Folder in dataFolder:

    (train_acc, valid_acc), (train_loss, valid_loss) = load_Npy(Folder)

    full_Plot(train_loss, valid_loss, train_acc, valid_acc, Folder)
