import os
import time
from matplotlib.pyplot import gray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import CycleGANConfig as config

class Gray_GanTrainer:
    def __init__(self,G,F,D_x,D_y,data_loader,
                 lambda_cycle=config.lambda_cycle,lambda_identity=config.lambda_identity,
                 use_initialization=False):
        # CycleGan Model
        self.G=G.to(config.device)
        self.F=F.to(config.device)
        self.D_x=D_x.to(config.device)
        self.D_y=D_y.to(config.device)
        
        # CycleGan optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config.lr, betas=(config.adam_beta1, 0.999))
        self.F_optimizer = optim.Adam(self.F.parameters(), lr=config.lr, betas=(config.adam_beta1, 0.999))
        self.D_x_optimizer = optim.Adam(self.D_x.parameters(), lr=config.lr, betas=(config.adam_beta1, 0.999))
        self.D_y_optimizer = optim.Adam(self.D_y.parameters(), lr=config.lr, betas=(config.adam_beta1, 0.999))
        
        # dataloader
        self.data_loader=data_loader

        # pool
        self.generated_x_images = ImagePool(50)
        self.generated_y_images = ImagePool(50)
                
        # criterion
        self.GAN_criterion=nn.MSELoss().to(config.device)
        self.Cycle_criterion=nn.L1Loss().to(config.device)
        self.Identity_criterion=nn.L1Loss().to(config.device)
        
        if use_initialization:
            self.Initialization_criterion = nn.L1Loss().to(config.device)
            self.lambda_initialization = config.content_loss_weight
        
        self.use_initialization = use_initialization

        self.curr_initialization_epoch = 0
        self.curr_epoch = 0

        self.init_loss_hist = []
        self.loss_D_x_hist = []
        self.loss_D_y_hist = []
        self.loss_G_GAN_hist = []
        self.loss_F_GAN_hist = []
        self.loss_cycle_hist = []
        self.loss_identity_hist = []
        self.print_every = config.print_every

    def train(self, num_epochs=config.num_epochs, initialization_epochs=0, save_path="checkpoints/CycleGAN/"):
    
        for init_epoch in range(self.curr_initialization_epoch, initialization_epochs):
            start = time.time()
            epoch_loss = 0

            for ix,(img,gray_img) in enumerate(self.data_loader):
                img = img.to(config.device)
                gray_img = gray_img.to(config.device)

                loss = self.initialize_step(img, gray_img)
                self.init_loss_hist.append(loss)
                epoch_loss += loss

                # print progress
                if (ix + 1) % self.print_every == 0:
                    print("Initialization Phase Epoch {0} Iteration {1}: Content Loss: {2:.4f}".format(init_epoch + 1,
                                                                                                       ix + 1,
                                                                                                       epoch_loss / (
                                                                                                                   ix + 1)))

            print("Initialization Phase [{0}/{1}], {2:.4f} seconds".format(init_epoch + 1, initialization_epochs,
                                                                           time.time() - start))
            self.curr_initialization_epoch += 1

        for epoch in range(self.curr_epoch, num_epochs):
            start = time.time()
            epoch_loss_D_x = 0
            epoch_loss_D_y = 0
            epoch_loss_G_GAN = 0
            epoch_loss_F_GAN = 0
            epoch_loss_cycle = 0
            epoch_loss_identity = 0

            for ix,(img,gray_img) in enumerate(self.data_loader):

                img = img.to(config.device)
                gray_img = gray_img.to(config.device)
                
                # step
                loss_D_x, loss_D_y, loss_G_GAN, loss_F_GAN, loss_cycle, loss_identity = self.train_step(img,
                                                                                                        gray_img)
                # hist
                self.loss_D_x_hist.append(loss_D_x.detach().item())
                self.loss_D_y_hist.append(loss_D_y.detach().item())
                self.loss_G_GAN_hist.append(loss_G_GAN.detach().item())
                self.loss_F_GAN_hist.append(loss_F_GAN.detach().item())
                self.loss_cycle_hist.append(loss_cycle.detach().item())
                self.loss_identity_hist.append(loss_identity.detach().item())

                epoch_loss_D_x += loss_D_x.detach().item()
                epoch_loss_D_y += loss_D_y.detach().item()
                epoch_loss_G_GAN += loss_G_GAN.detach().item()
                epoch_loss_F_GAN += loss_F_GAN.detach().item()
                epoch_loss_cycle += loss_cycle.detach().item()
                epoch_loss_identity += loss_identity.detach().item()
                # print progress
                if (ix + 1) % self.print_every == 0:
                    print("Training Phase Epoch {0} Iteration {1}: loss_D_x: {2:.4f} loss_D_y: {3:.4f} loss_G: {4:.4f} loss_F: {5:.4f} "
                          "loss_cycle: {6:.4f} loss_identity: {7:.4f}".format(epoch + 1, ix+1, epoch_loss_D_x / (ix + 1), epoch_loss_D_y / (ix + 1),
                                                                              epoch_loss_G_GAN / (ix + 1), epoch_loss_F_GAN / (ix + 1),
                                                                              epoch_loss_cycle / (ix + 1), epoch_loss_identity / (ix + 1)))  # print progress

            self.curr_epoch += 1
            print("Training Phase [{0}/{1}], {2:.4f} seconds".format(self.curr_epoch, num_epochs, time.time() - start))

        # Training finished, save checkpoint
        self.save_checkpoint(os.path.join(save_path, 'checkpoint-epoch-{0}.ckpt'.format(num_epochs)))

        return self.loss_D_x_hist, self.loss_D_y_hist, self.loss_G_GAN_hist, self.loss_F_GAN_hist, \
               self.loss_cycle_hist, self.loss_identity_hist

    def train_step(self, img, gray_img):
        # photo images are X, animation images are Y

        self.D_x_optimizer.zero_grad()
        self.D_y_optimizer.zero_grad()
        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()

        # Generate images and save them to image buffers
        print(img.size())
        generated_y = self.G(img)
        self.generated_y_images.save(generated_y.detach())

        generated_x = self.F(gray_img)
        self.generated_x_images.save(generated_x.detach())

        # train D_y with gray_images and generated_y
        gray_output = self.D_y(gray_img)
        gray_target = torch.ones_like(gray_output)
        loss_animation = self.GAN_criterion(gray_output, gray_target)
        loss_D_y = loss_animation

        generated_y_sample = self.generated_y_images.sample()
        generated_output = self.D_y(generated_y_sample)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.GAN_criterion(generated_output, generated_target)
        loss_D_y += loss_generated

        # according to cyclegan paper section 7.1, gradient of discriminators were divided by 2
        # to slow down the rate at which D learns relative G
        (loss_D_y / 2).backward()
        self.D_y_optimizer.step()

        # train D_x with img and generated_x
        photo_output = self.D_x(img)
        photo_target = torch.ones_like(photo_output)
        loss_photo = self.GAN_criterion(photo_output, photo_target)
        loss_D_x = loss_photo

        generated_x_sample = self.generated_x_images.sample()
        generated_output = self.D_x(generated_x_sample)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.GAN_criterion(generated_output, generated_target)
        loss_D_x += loss_generated

        # according to cyclegan paper section 7.1, gradient of discriminators were divided by 2
        # to slow down the rate at which D learns relative G
        (loss_D_x / 2).backward()
        self.D_x_optimizer.step()

        # time to train G and F
        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()

        # 1. GAN loss
        generated_y_output = self.D_y(generated_y)
        generated_y_target = torch.ones_like(generated_y_output)
        loss_G_GAN = self.GAN_criterion(generated_y_output, generated_y_target)

        generated_x_output = self.D_x(generated_x)
        generated_x_target = torch.ones_like(generated_x_output)
        loss_F_GAN = self.GAN_criterion(generated_x_output, generated_x_target)

        # 2. Cycle-Consistency loss
        cycle_x = self.F(generated_y)  # X -> Y -> X
        cycle_y = self.G(generated_x)  # Y -> X -> Y

        loss_cycle = self.lambda_cycle * self.Cycle_criterion(cycle_x, img)
        loss_cycle += self.lambda_cycle * self.Cycle_criterion(cycle_y, gray_img)

        # 3. identity loss
        G_y = self.G(gray_img)
        F_x = self.F(img)
        loss_identity = self.lambda_identity * self.Identity_criterion(G_y, gray_img)
        loss_identity += self.lambda_identity * self.Identity_criterion(F_x, img)

        generator_losses = loss_G_GAN + loss_F_GAN + loss_cycle + loss_identity
        generator_losses.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()

        return loss_D_x.detach().item(), loss_D_y.detach().item(), loss_G_GAN.detach().item(), loss_F_GAN.detach().item(), \
               loss_cycle.detach().item(), loss_identity.detach().item()

    def initialize_step(self, img, gray_img):
        # TODO
        # Train only using cycle-consistency and identity loss
        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()

        generated_y = self.G(img)
        generated_x = self.F(gray_img)

        cycle_x = self.F(generated_y)  # X -> Y -> X
        cycle_y = self.G(generated_x)  # Y -> X -> Y

        loss_cycle = self.lambda_cycle * self.Cycle_criterion(cycle_x, img)
        loss_cycle += self.lambda_cycle * self.Cycle_criterion(cycle_y, gray_img)

        G_y = self.G(gray_img)
        F_x = self.F(img)
        loss_identity = self.lambda_identity * self.Identity_criterion(G_y, gray_img)
        loss_identity += self.lambda_identity * self.Identity_criterion(F_x, img)

        initialization_loss = loss_cycle + loss_identity
        initialization_loss.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()

        return initialization_loss.detach().item()

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                'G_state_dict': self.G.state_dict(),
                'F_state_dict': self.F.state_dict(),
                'D_x_state_dict': self.D_x.state_dict(),
                'D_y_state_dict': self.D_y.state_dict(),
                'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                'F_optimizer_state_dict': self.F_optimizer.state_dict(),
                'D_x_optimizer_state_dict': self.D_x_optimizer.state_dict(),
                'D_y_optimizer_state_dict': self.D_y_optimizer.state_dict(),
                'loss_D_x_hist': self.loss_D_x_hist,
                'loss_D_y_hist': self.loss_D_y_hist,
                'loss_G_GAN_hist': self.loss_G_GAN_hist,
                'loss_F_GAN_hist': self.loss_F_GAN_hist,
                'loss_cycle_hist': self.loss_cycle_hist,
                'loss_identity_hist': self.loss_identity_hist,
                'init_loss_hist': self.init_loss_hist,
                'curr_epoch': self.curr_epoch

            }, checkpoint_path
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.F.load_state_dict(checkpoint['F_state_dict'])
        self.D_x.load_state_dict(checkpoint['D_x_state_dict'])
        self.D_y.load_state_dict(checkpoint['D_y_state_dict'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.F_optimizer.load_state_dict(checkpoint['F_optimizer_state_dict'])
        self.D_x_optimizer.load_state_dict(checkpoint['D_x_optimizer_state_dict'])
        self.D_y_optimizer.load_state_dict(checkpoint['D_y_optimizer_state_dict'])
        self.loss_D_x_hist = checkpoint['loss_D_x_hist']
        self.loss_D_y_hist = checkpoint['loss_D_y_hist']
        self.loss_G_GAN_hist = checkpoint['loss_G_GAN_hist']
        self.loss_F_GAN_hist = checkpoint['loss_F_GAN_hist']
        self.loss_cycle_hist = checkpoint['loss_cycle_hist']
        self.loss_identity_hist = checkpoint['loss_identity_hist']
        self.init_loss_hist = checkpoint['init_loss_hist']
        try:
            self.curr_epoch = checkpoint['curr_epoch']
        except:
            self.curr_epoch = int(checkpoint_path.split('-')[-1].split('.')[0])

class ImagePool:
    
    def __init__(self, maxlen=50):
        self.buffer = []
        self.maxlen = maxlen

    def save(self, images):
        for image in images:
            if len(self.buffer) >= self.maxlen:
                idx = np.random.randint(0, self.maxlen)
                self.buffer[idx] = image
            else:
                self.buffer.append(image)

    def sample(self, sample_size=config.batch_size):
        idxs = np.random.choice(len(self.buffer), sample_size)
        return torch.stack([self.buffer[idx] for idx in idxs])

