import torch
from torch.autograd import Variable
import numpy as np

START_TRAIN_AT_IMG_SIZE = 4
CHECKPOINT_GEN = "Models_saved/generator_4.pth"
CHECKPOINT_CRITIC = "Models_saved/discriminator_4.pth"
RESULTS_DIR="Models_saved"
PATH_CSV_SICAP='Z:\Proyectos\SICAP\INVESTIGACION\DATASETS\SICAP\processed\SICAPv2\partition\Test\Train.xlsx'
PATH_IMAGES_SICAP='Z:\Proyectos\SICAP\INVESTIGACION\DATASETS\SICAP\processed\SICAPv2\images'
cuda = True if torch.cuda.is_available() else False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE_GENERATOR = 1e-3
LEARNING_RATE_DISCRIMINATOR =1e-3
BATCH_SIZES = [64, 64, 64, 64, 64, 64, 32, 16, 4]
CHANNELS_IMG = 3
Z_DIM = 512
IN_CHANNELS = 512
CRITIC_ITERATIONS = 1
LAMBDA_GP = 5
PROGRESSIVE_EPOCHS = [2] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
N_CLASSES=4
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
FIXED_LABELS=Variable(LongTensor(np.random.randint(0, N_CLASSES, 8)))
NUM_WORKERS = 4