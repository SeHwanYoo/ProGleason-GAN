import torch

START_TRAIN_AT_IMG_SIZE = 4
CHECKPOINT_GEN = "Models_saved/generator_4.pth"
CHECKPOINT_CRITIC = "Models_saved/discriminator_4.pth"
RESULTS_DIR="Models_saved"
PATH_CSV_SICAP='\SICAPv2\partition\Test\Train.xlsx'
# PATH_IMAGES_SICAP='\SICAPv2\images'
PATH_IMAGES_SICAP='/home/yoos-bii/Desktop/data_tct/prostate_output'
cuda = True if torch.cuda.is_available() else False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE_GENERATOR = 1e-3
LEARNING_RATE_DISCRIMINATOR =1e-3
BATCH_SIZES = [64, 64, 64, 64, 64, 64, 32, 16, 4]
CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 5
PROGRESSIVE_EPOCHS = [100] * len(BATCH_SIZES)
# PROGRESSIVE_EPOCHS = [1] * len(BATCH_SIZES)
N_CLASSES=6
NUM_WORKERS = 4