import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSet
from data.simple_dataset import SimpleDataSet

from model.VGDCFusion_model import VGDCFusion as create_model
from scripts.utils import read_data, train_one_epoch, create_lr_scheduler
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T

import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from torchvision.transforms import ToTensor

def normalize_img(img_tensor):
    img = img_tensor[0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def add_labels_and_log(writer, images, labels, tag, epoch):
    # images: list of tensors (C, H, W)
    # labels: list of str
    fig, axs = plt.subplots(1, len(images), figsize=(3 * len(images), 3))

    for i, (img, label) in enumerate(zip(images, labels)):
        img_np = img.cpu().numpy()
        if img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)  
            axs[i].imshow(img_np, cmap='gray')
        else:
            img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W -> H,W,C
            axs[i].imshow(img_np)
        axs[i].set_title(label)
        axs[i].axis('off')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)
    writer.add_image(tag, image, epoch)
    plt.close(fig)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/Text_train_{}".format(file_name)
    os.makedirs(filefold_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_loss = 1e5
    start_epoch = 0

    print("Loading IVF Fusion and Low-Light & Low Contrast Task!")
    if args.lowlight_contrast_path is not None:
        train_lowlight_contrast_path_list = read_data(args.lowlight_contrast_path)
    else:
        train_lowlight_contrast_path_list = None

    print("Loading IVF Fusion and Low-Light & Noise Task!")
    if args.lowlight_noise_path is not None:
        train_lowlight_noise_path_list = read_data(args.lowlight_noise_path)
    else:
        train_lowlight_noise_path_list = None

    print("Loading IVF Fusion and Overexposure & Low Contrast Task!")
    if args.overexposure_contrast_path is not None:
        train_overexposure_contrast_path_list = read_data(args.overexposure_contrast_path)
    else:
        train_overexposure_contrast_path_list = None

    print("Loading IVF Fusion and Overexposure & Noise Task!")
    if args.overexposure_noise_path is not None:
        train_overexposure_noise_path_list = read_data(args.overexposure_noise_path)
    else:
        train_overexposure_noise_path_list = None

    data_transform = {
        "train": T.Compose([T.RandomCrop(96),#patch_size
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = PromptDataSet(train_lowlight_contrast_path_list=train_lowlight_contrast_path_list,
                                  train_lowlight_noise_path_list=train_lowlight_noise_path_list,
                                  train_overexposure_contrast_path_list=train_overexposure_contrast_path_list,
                                  train_overexposure_noise_path_list=train_overexposure_noise_path_list,
                                  phase="train",
                                  transform=data_transform["train"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    model_clip, _ = clip.load("ViT-B/32", device=device)

    model = create_model(model_clip).to(device)

    for param in model.model_clip.parameters():
        param.requires_grad = False

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_max_loss, train_color_loss, train_text_loss, lr, I_A, I_B, I_A_gt, I_B_gt, I_fused = train_one_epoch(model=model,
                                              model_clip=model_clip,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch)
        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_max_loss", train_max_loss, epoch)
        tb_writer.add_scalar("train_color_loss", train_color_loss, epoch)
        tb_writer.add_scalar("train_text_loss", train_text_loss, epoch)

        imgs = [
            normalize_img(I_A),
            normalize_img(I_B),
            normalize_img(I_A_gt),
            normalize_img(I_B_gt),
            normalize_img(I_fused)
        ]
        labels = ["Input:Visible", "Input:Infrared", "GT:Visible", "GT:Infrared", "Fused Output"]

        add_labels_and_log(tb_writer, imgs, labels, tag="ComparisonGrid", epoch=epoch)

        if train_loss < best_loss:
            if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
            best_loss = train_loss

        if args.use_dp == True:
            save_file = {"model": model.module.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
        else:
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
        torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)

    # set the appropriate batch-size value for your device
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00025)

    parser.add_argument('--lowlight_contrast_path', type=str, default="./dataset/Text_Train/VI_lowlight_IR_lowcontrast")
    parser.add_argument('--lowlight_noise_path', type=str, default="./dataset/Text_Train/VI_lowlight_IR_noise")
    parser.add_argument('--overexposure_contrast_path', type=str, default="./dataset/Text_Train/VI_overexposure_IR_lowcontrast")
    parser.add_argument('--overexposure_noise_path', type=str, default="./dataset/Text_Train/VI_overexposure_IR_noise")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default = False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()

    main(opt)
