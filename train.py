""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.optim as optim
from utils import (
    gradient_penalty_ACGAN,
    save_checkpoint,
    load_checkpoint,
    modify_loader,
    get_loader)

from models import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config as config
from torch.autograd import Variable
import os
import numpy as np

import matplotlib.pyplot as plt


import wandb

wandb.init(
    project="prostate_gan", 
    
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 10,
    # }
    )


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

torch.backends.cudnn.benchmarks = True


def train_fn(
        critic,
        gen,
        loader,
        dataset,
        step,
        alpha,
        opt_critic,
        opt_gen,
        scaler_gen,
        scaler_critic,
        epoch=10, 
        idx_num_epochs=0
            ):
    loop = tqdm(loader, leave=True)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    for batch_idx, (real, labels) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        labels = Variable(labels.type(LongTensor))

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, labels, alpha, step)
            critic_real = critic(real, labels,alpha, step)
            critic_fake = critic(fake.detach(), labels,alpha, step)

            gp = gradient_penalty_ACGAN(critic, real, fake, alpha, step, labels,device=config.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake= critic(fake, labels,alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)
        loop.set_description(f"[Loss Critic: {loss_critic.item():.4f}, Loss Generator : {loss_gen.item():.4f}] -- PROGRESION BATCH:")
        loop.update(1)
        
        wandb.log({"Loss Critic": loss_critic.item(), "Loss Generator": loss_gen.item()})

    
    
    if epoch % 50 == 0:
    # if True:
        gen.eval() 
        
        noise = torch.randn(config.N_CLASSES, config.Z_DIM, 1, 1).to(config.DEVICE)
        
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
        labels = torch.Tensor([0, 1, 2, 3, 4, 5])
        labels = Variable(labels.type(LongTensor))
        
        eval_fake = gen(noise, labels, alpha, step)
        # print('------------------------>', fake.shape)
        # print('------------------------>', type(fake))
        
        fake_images = eval_fake.detach().cpu() 
        fake_images = fake_images.permute(0, 2, 3, 1)
        fake_images = np.clip(fake_images, 0, 1)

        # idx = 0 
        for cls in range(config.N_CLASSES): 
            plt.subplot(1, config.N_CLASSES, cls+1)
            plt.imshow(fake_images[cls])
            plt.title(f'Class: {(str(cls))}')
            
        plt.savefig(os.path.join(config.RESULTS_DIR, f'STEP_{str(idx_num_epochs)}_EPOCH_{str(epoch)}_{str(np.random.randint(0, 1000))}'))
        wandb.log({"Log generator images": wandb.Image(os.path.join(config.RESULTS_DIR, f'STEP_{str(idx_num_epochs)}_EPOCH_{str(epoch)}_{str(np.random.randint(0, 1000))}'))})

        
        
    return  alpha


def main():
    gen = Generator(config.Z_DIM, n_classes=config.N_CLASSES,in_channels=config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, n_classes=config.N_CLASSES,img_channels=config.CHANNELS_IMG
                           ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()


    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GENERATOR,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE_DISCRIMINATOR,
        )


    dataset_loaded=0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    idx_num_epochs = 0
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        if(dataset_loaded==0):
            loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
            dataset_loaded=1
        else:
            loader=modify_loader(4*2**step,dataset)

        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            
            # moving, becuase inside of train funciton, eval mode 
            gen.train()
            critic.train()
            
            alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                scaler_gen,
                scaler_critic,
                epoch=epoch, # add
                idx_num_epochs = idx_num_epochs, # add
            )

        if config.SAVE_MODEL:
            if not os.path.exists(config.RESULTS_DIR):
                os.makedirs(config.RESULTS_DIR)
            save_checkpoint(gen, opt_gen, filename=os.path.join(config.RESULTS_DIR,'generator_' + str(step) + str('.pth')))
            save_checkpoint(critic, opt_critic, filename=os.path.join(config.RESULTS_DIR,'discriminator_' + str(step) + str('.pth')))
        step += 1  # progress to the next img size

if __name__ == "__main__":
    main()