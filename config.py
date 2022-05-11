import torch

class CycleGANConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataloader.py
    batch_size = 2
    num_workers = 4
    photo_image_dir = "data/afhq/train"
    test_photo_image_dir = "data/test/val"

    # CycleGAN_train.py
    lambda_cycle = 10.0  # lambda parameter for cycle loss, X -> Y -> X and Y -> X -> Y
    lambda_identity = 0.5  # lambda parameter for identity loss, helpful for image style transfer task

    adam_beta1 = 0.5  # following dcgan
    lr = 0.0002
    num_epochs = 10
    initialization_epochs = 0
    content_loss_weight = 10
    print_every = 100
