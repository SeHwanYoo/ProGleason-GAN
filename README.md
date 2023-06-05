# TRAINING INSTRUCTIONS

This module allows you to rebuild slides that lack resolution levels required for patching. **Please read the requirements 
section if you don't have the library *pyvips* installed**.

This is an example script call `python recontructor.py --p1 Directory_with_incomplete_slides --p2 Output_directory`.

| Arguments | Description |
|-----------|-------------|
|       --START_TRAIN_AT_IMG_SIZE    |       Path containing all the slides to reconstruct      |
|    --CHECKPOINT_GEN                            |     Output directory for the recontructed slides         |
|    --CHECKPOINT_CRITIC                           |     Output directory for the recontructed slides         |
|    --RESULTS_DIR                            |     Output directory for the recontructed slides         |
|    --PATH_CSV_SICAP                            |     Output directory for the recontructed slides         |
|    --PATH_IMAGES_SICAP                            |     Output directory for the recontructed slides         |
|    --DEVICE                            |     Output directory for the recontructed slides         |
|    --SAVE_MODEL                            |     Output directory for the recontructed slides         |
|    --LOAD_MODEL                            |     Output directory for the recontructed slides         |
|    --LEARNING_RATE_GENERATOR|     Output directory for the recontructed slides         |
|    --LEARNING_RATE_DISCRIMINATOR|     Output directory for the recontructed slides         |
|    --BATCH_SIZES|     Output directory for the recontructed slides         |
|    --CHANNELS_IMG|     Output directory for the recontructed slides         |
|    --Z_DIM|     Output directory for the recontructed slides         |
|    --IN_CHANNELS|     Output directory for the recontructed slides         |
|    --LAMBDA_GP|     Output directory for the recontructed slides         |
|    --PROGRESSIVE_EPOCHS|     Output directory for the recontructed slides         |
|    --N_CLASSES|     Output directory for the recontructed slides         |
|    --NUM_WORKERS|     Output directory for the recontructed slides         |



This is an example script call `python train.py'