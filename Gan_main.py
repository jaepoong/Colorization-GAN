import wandb
from core.Generators import Generator,Generator_Mod,Generator2
from core.Discriminator import Discriminator,Discriminator_Mod
from Gan_train import Gray_GanTrainer
from config import CycleGANConfig as config
from core.data_loader import Gray_RGB_dataset,get_gray_train_loader,get_gray_test_loader, get_train_loader,get_test_loader

import torch
import argparse
import torchvision.utils as tvutils
import os
from torchvision import transforms

def load_model(G, F, D_x, D_y, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    F.load_state_dict(checkpoint['F_state_dict'])
    D_y.load_state_dict(checkpoint['D_y_state_dict'])
    D_x.load_state_dict(checkpoint['D_x_state_dict'])


def load_generators(G, F, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    F.load_state_dict(checkpoint['F_state_dict'])


def generate_and_save_images(generator, test_image_loader, save_path):
    # for each image in test_image_loader, generate image and save
    generator.eval()
    torch_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])

    image_ix = 0
    for test_images in test_image_loader:
        test_images = test_images.to(config.device)
        generated_images = generator(test_images).detach().cpu()

        for i in range(len(generated_images)):
            image = generated_images[i]
            image = torch_to_image(image)
            image.save(os.path.join(save_path, '{0}.jpg'.format(image_ix)))
            image_ix += 1
        break

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test',
                        action='store_true',
                        help='Use this argument to test generator and compute FID score')
    parser.add_argument('--cartoonizing',
                        default=config.cartoonizing,
                        help="if use cartoonizing make it true")
    
    parser.add_argument('--Mod',
                        default=config.mod)

    parser.add_argument('--model_path',
                        help='Path to saved model')

    parser.add_argument('--model_save_path',
                        default='checkpoints/CycleGAN/',
                        help='path to save checkpoint when training is finished.')

    parser.add_argument("--photo_image_dir",
                        default=config.photo_image_dir,
                        help="Path to photo images")
    
    parser.add_argument("--photo_image_target_dir",
                        default=config.photo_image_target_dir,
                        help="Path to photo images")

    parser.add_argument('--test_image_path',
                        default=config.test_image_path,
                        help='Path to test photo images')
    
    parser.add_argument('--image_test',
                        default=True,
                        help="epoch 단위로 이미지 시각화 할거?")
    
    parser.add_argument('--generated_image_save_path',
                        default='generated_images/CycleGAN/',
                        help='path to save generated images')

    parser.add_argument('--initialization_epochs',
                        type=int,
                        default=config.initialization_epochs,
                        help='Number of epochs for initialization phase')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=config.num_epochs,
                        help='Number of training epochs')

    parser.add_argument("--batch_size",
                        type=int,
                        default=config.batch_size)

    parser.add_argument('--use_edge_smoothed_images',
                        action='store_true',
                        help='Use this argument to use edge smoothed images in training')

    parser.add_argument('--test_animation_to_photo',
                        default=False,
                        help='Use this argument to test animation to photo transfer')

    parser.add_argument('--load_data_on_ram',
                        action='store_true',
                        help="Use this argument to load entire dataset on ram. (useful in AWS setting)")

    parser.add_argument('--use_bias',
                        default=False,
                        help="Use this argument to load entire dataset on ram. (useful in AWS setting)")
    args = parser.parse_args()

    return args


def main():
    
    args = get_args()

    device = config.device
    print("PyTorch running with device {0}".format(device))

    if args.test:
        assert args.model_path, 'model_path must be provided for testing'
        print('Testing...')

        print("Creating models...")
        if args.Mod:
            G = Generator_Mod(use_bias=args.use_bias).to(device)
            F = Generator_Mod(use_bias=args.use_bias).to(device)
            G.eval()
            F.eval()
        else:
            G=Generator(use_bias=args.use_bias).to(device)
            F=Generator(use_bias=args.use_bias).to(device)
            G.eval()
            F.eval()
        print('Loading models...')
        load_generators(G, F, args.model_path)

        if not args.test_animation_to_photo:
            generator = G
        else:
            generator = F
        if args.cartoonizing:
            test_images = get_test_loader(root=args.test_image_path, batch_size=config.batch_size, shuffle=False)
        else:    
            test_images = get_gray_test_loader(root=args.test_image_path, batch_size=config.batch_size, shuffle=False)
        print(test_images)
        image_batch= next(iter(test_images))
        image_batch = image_batch.to(config.device)

        new_images = G(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))
        tvutils.save_image(new_images, 'generated_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))

        if not os.path.isdir('generated_images/CycleGAN'):
            os.makedirs('generated_images/CycleGAN/')

        generate_and_save_images(generator, test_images, args.generated_image_save_path)

    else:
        print("Training...")

        print("Loading 2 generators and 2 discriminators")
        print("Modifided Trainiing : ",args.Mod)
        
        if args.Mod:
            G = Generator_Mod(use_bias=args.use_bias).to(device)
            F = Generator_Mod(use_bias=args.use_bias).to(device)
            D_x = Discriminator_Mod(use_bias=args.use_bias).to(device)
            D_y = Discriminator_Mod(use_bias=args.use_bias).to(device)
        
        else:
            G = Generator2(use_bias=args.use_bias).to(device)
            F = Generator2(use_bias=args.use_bias).to(device)
            D_x = Discriminator(use_bias=args.use_bias).to(device)
            D_y = Discriminator(use_bias=args.use_bias).to(device)            

        # load dataloaders
        if not args.cartoonizing:
            loader,target_loader = get_gray_train_loader(root=args.photo_image_dir,
                                       target_root=args.photo_image_target_dir, batch_size=args.batch_size)
        else:
            loader,target_loader = get_train_loader(root=args.photo_image_dir,
                                       target_root=args.photo_image_target_dir, batch_size=args.batch_size)
        
            

        trainer = Gray_GanTrainer(G, F, D_x, D_y, loader,target_loader ,args.generated_image_save_path,
                                  use_initialization=(args.initialization_epochs > 0),mod=args.Mod,image_test=args.image_test)
        if args.model_path:
            trainer.load_checkpoint(args.model_path)

        print('Start Training...')

        loss_D_x_hist, loss_D_y_hist, loss_G_GAN_hist, loss_F_GAN_hist, \
        loss_cycle_hist, loss_identity_hist = trainer.train(num_epochs=args.num_epochs,
                                                            initialization_epochs=args.initialization_epochs,
                                                            save_path=args.model_save_path)
        # 시험용으로 해봄
        if not args.cartoonizing:
            test_images = get_gray_test_loader(root=args.test_image_path, batch_size=config.batch_size, shuffle=False)
        else:
            test_images = get_test_loader(root=args.test_image_path, batch_size=config.batch_size, shuffle=False)
        print(test_images)
        image_batch= next(iter(test_images))
        image_batch = image_batch.to(config.device)

        new_images = G(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))
        tvutils.save_image(new_images, 'generated_images.jpg', nrow=3, padding=2, normalize=True, value_range=(-1, 1))

if __name__ == '__main__':
    main()
