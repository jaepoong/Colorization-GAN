import torch

class CycleGANConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mod=False
    # dataloader.py
    cartoonizing=False
    batch_size = 6
    num_workers = 4
    photo_image_dir = "data/afhq/train"
    photo_image_target_dir="data/afhq/train"
    test_image_path = "data/afhq/val"
    model_path=None
    # CycleGAN_train.py
    lambda_cycle = 10.0  # lambda parameter for cycle loss, X -> Y -> X and Y -> X -> Y
    lambda_identity = 0.5  # lambda parameter for identity loss, helpful for image style transfer task

    adam_beta1 = 0.5  # following dcgan
    lr = 0.0002
    num_epochs = 50
    initialization_epochs = 10
    content_loss_weight = 5
    print_every = 100