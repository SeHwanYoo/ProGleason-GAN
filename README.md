# INFORMATION

This is the code implementation for: "ProGleason-GAN: Conditional Progressive Growing: GAN for prostatic cancer Gleason Grade patch synthesis"

# TRAINING INSTRUCTIONS

To initiate the training, the following parameters must be set in the config.py file.

| Arguments | Description |
|-----------|-------------|
|       --START_TRAIN_AT_IMG_SIZE    |       Start resolution      |
|    --CHECKPOINT_GEN                            |     Path for generator checkpoint         |
|    --CHECKPOINT_CRITIC                           |     Path for discriminator checkpoint         |
|    --RESULTS_DIR                            |     Output directory for the recontructed slides         |
|    --PATH_CSV_SICAP                            |     Path with SICAPv2 partition annotations         |
|    --PATH_IMAGES_SICAP                            |     Path containing SICAPv2 patches         |
|    --DEVICE                            |     DEVICE INFO (cpu or cuda)        |
|    --SAVE_MODEL                            |     Flag to allow the training to save the model in the RESULTS_DIR         |
|    --LOAD_MODEL                            |     Flag to allow the training to load previous checkpoints         |
|    --LEARNING_RATE_GENERATOR|     Learning rate for the generator model         |
|    --LEARNING_RATE_DISCRIMINATOR|     Learning rate for the discriminator model        |
|    --BATCH_SIZES|     List of batch sizes for each resolution         |
|    --CHANNELS_IMG|     The number of channels in the input images         |
|    --Z_DIM|     Size of the input noise vector         |
|    --IN_CHANNELS|     The number of channels in the generator's input noise vector        |
|    --LAMBDA_GP|     The weight factor for the gradient penalty term used in the Wasserstein GAN (WGAN) loss         |
|    --PROGRESSIVE_EPOCHS|     List of training epochs for each resolution         |
|    --N_CLASSES|     Number of classes in the dataset         |
|    --NUM_WORKERS|     The number of parallel workers for data loading during training         |


After that, you only need to call
```
$ python train.py
``` 