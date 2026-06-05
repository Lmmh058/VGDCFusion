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

lowlight_contrast_prompt_ir_path = "./dataset/Text_Train/VI_lowlight_IR_lowcontrast/train/text_ir.txt"
assert os.path.exists(lowlight_contrast_prompt_ir_path), "text prompt root: {} does not exist.".format(lowlight_contrast_prompt_ir_path)
with open(lowlight_contrast_prompt_ir_path, 'r', encoding='utf-8') as file:
        lowlight_contrast_lines_ir = file.readlines()

lowlight_contrast_prompt_vi_path = "./dataset/Text_Train/VI_lowlight_IR_lowcontrast/train/text_vi.txt"
assert os.path.exists(lowlight_contrast_prompt_vi_path), "text prompt root: {} does not exist.".format(lowlight_contrast_prompt_vi_path)
with open(lowlight_contrast_prompt_vi_path, 'r', encoding='utf-8') as file:
        lowlight_contrast_lines_vi = file.readlines()

lowlight_noise_prompt_ir_path = "./dataset/Text_Train/VI_lowlight_IR_noise/train/text_ir.txt"
assert os.path.exists(lowlight_noise_prompt_ir_path), "text prompt root: {} does not exist.".format(lowlight_noise_prompt_ir_path)
with open(lowlight_noise_prompt_ir_path, 'r', encoding='utf-8') as file:
        lowlight_noise_lines_ir = file.readlines()

lowlight_noise_prompt_vi_path = "./dataset/Text_Train/VI_lowlight_IR_noise/train/text_vi.txt"
assert os.path.exists(lowlight_noise_prompt_vi_path), "text prompt root: {} does not exist.".format(lowlight_noise_prompt_vi_path)
with open(lowlight_noise_prompt_vi_path, 'r', encoding='utf-8') as file:
        lowlight_noise_lines_vi = file.readlines()

overexposure_contrast_prompt_ir_path = "./dataset/Text_Train/VI_overexposure_IR_lowcontrast/train/text_ir.txt"
assert os.path.exists(overexposure_contrast_prompt_ir_path), "text prompt root: {} does not exist.".format(overexposure_contrast_prompt_ir_path)
with open(overexposure_contrast_prompt_ir_path, 'r', encoding='utf-8') as file:
        overexposure_contrast_lines_ir = file.readlines()

overexposure_contrast_prompt_vi_path = "./dataset/Text_Train/VI_overexposure_IR_lowcontrast/train/text_vi.txt"
assert os.path.exists(overexposure_contrast_prompt_vi_path), "text prompt root: {} does not exist.".format(overexposure_contrast_prompt_vi_path)
with open(overexposure_contrast_prompt_vi_path, 'r', encoding='utf-8') as file:
        overexposure_contrast_lines_vi = file.readlines()

overexposure_noise_prompt_ir_path = "./dataset/Text_Train/VI_overexposure_IR_noise/train/text_ir.txt"
assert os.path.exists(overexposure_noise_prompt_ir_path), "text prompt root: {} does not exist.".format(overexposure_noise_prompt_ir_path)
with open(overexposure_noise_prompt_ir_path, 'r', encoding='utf-8') as file:
        overexposure_noise_lines_ir = file.readlines()

overexposure_noise_prompt_vi_path = "./dataset/Text_Train/VI_overexposure_IR_noise/train/text_vi.txt"
assert os.path.exists(overexposure_noise_prompt_vi_path), "text prompt root: {} does not exist.".format(overexposure_noise_prompt_vi_path)
with open(overexposure_noise_prompt_vi_path, 'r', encoding='utf-8') as file:
        overexposure_noise_lines_vi = file.readlines()

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root= os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root= os.path.join(train_root, "Infrared_gt")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))

    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path=train_visible_gt_path[index]
        img_infrared_gt_path=train_infrared_gt_path[index]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)


    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(train_infrared_gt_path)

    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))


    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]

    return train_low_light_path_list

def get_lowlight_contrast_prompt():
    random_line_ir = random.choice(lowlight_contrast_lines_ir)
    random_line_ir = random_line_ir.strip()
    random_line_vi = random.choice(lowlight_contrast_lines_vi)
    random_line_vi = random_line_vi.strip()
    return random_line_ir, random_line_vi

def get_lowlight_noise_prompt():
    random_line_ir = random.choice(lowlight_noise_lines_ir)
    random_line_ir = random_line_ir.strip()
    random_line_vi = random.choice(lowlight_noise_lines_vi)
    random_line_vi = random_line_vi.strip()
    return random_line_ir, random_line_vi

def get_overexposure_contrast_prompt():
    random_line_ir = random.choice(overexposure_contrast_lines_ir)
    random_line_ir = random_line_ir.strip()
    random_line_vi = random.choice(overexposure_contrast_lines_vi)
    random_line_vi = random_line_vi.strip()
    return random_line_ir, random_line_vi

def get_overexposure_noise_prompt():
    random_line_ir = random.choice(overexposure_noise_lines_ir)
    random_line_ir = random_line_ir.strip()
    random_line_vi = random.choice(overexposure_noise_lines_vi)
    random_line_vi = random_line_vi.strip()
    return random_line_ir, random_line_vi

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

    last_I_A = last_I_B = last_I_A_gt = last_I_B_gt = last_I_fused = None

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line_ir = []
        text_line_vi = []

        for index in range(len(task)):
        # default type degradation in vis image
            if task[index] == "lowlight_contrast":
                text_ir, text_vi = get_lowlight_contrast_prompt()
                text_line_ir.append(text_ir)
                text_line_vi.append(text_vi)
            elif task[index] == "lowlight_noise":
                text_ir, text_vi = get_lowlight_noise_prompt()
                text_line_ir.append(text_ir)
                text_line_vi.append(text_vi)
            elif task[index] == "overexposure_contrast":
                text_ir, text_vi = get_overexposure_contrast_prompt()
                text_line_ir.append(text_ir)
                text_line_vi.append(text_vi)
            elif task[index] == "overexposure_noise":
                text_ir, text_vi = get_overexposure_noise_prompt()
                text_line_ir.append(text_ir)
                text_line_vi.append(text_vi)
            else:
                text_line_ir.append("This is unknown to the image fusion task.")
                text_line_vi.append("This is unknown to the image fusion task.")
                print("Warning! Task Undefined!")

        text_clip_ir = clip.tokenize(text_line_ir).to(device)
        text_clip_vi = clip.tokenize(text_line_vi).to(device)

        if torch.cuda.is_available():
            #B,C,H,W
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        I_fused = model(I_A, I_B, text_clip_ir, text_clip_vi)

        loss, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        # accu_ssim_loss += loss_ssim.detach()
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
        last_I_A_gt = I_A_gt.detach().cpu()
        last_I_B_gt = I_B_gt.detach().cpu()
        last_I_fused = I_fused.detach().cpu()

    return (accu_total_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            lr,last_I_A, last_I_B, last_I_A_gt, last_I_B_gt, last_I_fused)


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
