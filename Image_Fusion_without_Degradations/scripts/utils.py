import os
import sys
import random
import clip

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.losses import fusion_prompt_loss

IVIF_prompt_path = "./Train/text.txt"
assert os.path.exists(IVIF_prompt_path), "text prompt root: {} does not exist.".format(IVIF_prompt_path)
with open(IVIF_prompt_path, 'r', encoding='utf-8') as file:
        IVIF_lines = file.readlines()

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = root
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)


    train_images_visible_path = []
    train_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  

    train_visible_root = os.path.join(train_root, "vi")
    train_infrared_root= os.path.join(train_root, "ir")


    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]


    train_visible_path.sort()
    train_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))

    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path)

    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))

    return train_images_visible_path, train_images_infrared_path

def get_IVIF_prompt():
    random_line = random.choice(IVIF_lines)
    random_line = random_line.strip()
    return random_line

def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_clip.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    last_I_A = last_I_B = last_I_fused = None

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, _ = data
        text_line_ir = []
        text_line_vi = []

        for _ in range(I_A.shape[0]):  # batch size
            prompt_ir = get_IVIF_prompt()
            prompt_vi = prompt_ir
            text_line_ir.append(prompt_ir)
            text_line_vi.append(prompt_vi)

        text_clip_ir = clip.tokenize(text_line_ir).to(device)
        text_clip_vi = clip.tokenize(text_line_vi).to(device)

        if torch.cuda.is_available():
            #B,C,H,W
            I_A = I_A.to(device)
            I_B = I_B.to(device)

        I_fused = model(I_A, I_B, text_clip_ir, text_clip_vi)

        loss, loss_max, loss_color, loss_text = loss_function_prompt(I_A, I_B, I_fused)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch,
                         accu_total_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Save the last batch's input and output
        last_I_A = I_A.detach().cpu()
        last_I_B = I_B.detach().cpu()
        last_I_fused = I_fused.detach().cpu()

    return (accu_total_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            lr,last_I_A, last_I_B, last_I_fused)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
