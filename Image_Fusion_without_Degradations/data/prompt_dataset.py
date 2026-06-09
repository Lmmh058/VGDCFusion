from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

class PromptDataSet(Dataset):
    def __init__(self, train_lowlight_contrast_path_list, train_lowlight_noise_path_list,
                 train_overexposure_contrast_path_list, train_overexposure_noise_path_list, phase="train",transform=None):
        self.phase = phase
        if phase == "train":
            self.paths = {
                'lowlight_contrast_A': train_lowlight_contrast_path_list[0],
                'lowlight_contrast_B': train_lowlight_contrast_path_list[1],

                'lowlight_noise_A': train_lowlight_noise_path_list[0],
                'lowlight_noise_B': train_lowlight_noise_path_list[1],

                'overexposure_contrast_A': train_overexposure_contrast_path_list[0],
                'overexposure_contrast_B': train_overexposure_contrast_path_list[1],

                'overexposure_noise_A': train_overexposure_noise_path_list[0],
                'overexposure_noise_B': train_overexposure_noise_path_list[1],
            }
            self.paths_gt = {
                'lowlight_contrast_A_gt': train_lowlight_contrast_path_list[2],
                'lowlight_contrast_B_gt': train_lowlight_contrast_path_list[3],

                'lowlight_noise_A_gt': train_lowlight_noise_path_list[2],
                'lowlight_noise_B_gt': train_lowlight_noise_path_list[3],

                'overexposure_contrast_A_gt': train_overexposure_contrast_path_list[2],
                'overexposure_contrast_B_gt': train_overexposure_contrast_path_list[3],

                'overexposure_noise_A_gt': train_overexposure_noise_path_list[2],
                'overexposure_noise_B_gt': train_overexposure_noise_path_list[3],
            }
        else:
            
            raise NotImplementedError("Wrong Phase")
        self.transform = transform

        # Create a list to hold all sample indices grouped by class
        self.class_indices = {}
        for class_key, paths in self.paths.items():
            self.class_indices[class_key] = list(range(len(paths)))
        pass

    def __len__(self):
        if self.phase == "train":
            return sum(len(paths) for paths in self.paths.values())
        else:
            return 80

    def __getitem__(self, item):
        # Randomly select a class, use the random sampling (equal to sequential sampling when the number of sampling is large)
        
        class_key = random.choice(list(self.paths.keys()))

        # Randomly select an index for the chosen class
        class_indices = self.class_indices[class_key]
        item_index = random.randint(0, len(class_indices) - 1)
        image_index = class_indices[item_index]

        # Load the A and B images based on the class and index
        image_A_path = self.paths[class_key[:-2] + '_A'][image_index]
        image_B_path = self.paths[class_key[:-2] + '_B'][image_index]

        image_A_gt_path = self.paths_gt[class_key[:-2] + '_A_gt'][image_index]
        image_B_gt_path = self.paths_gt[class_key[:-2] + '_B_gt'][image_index]

        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')
        image_A_gt = Image.open(image_A_gt_path).convert(mode='RGB')
        image_B_gt = Image.open(image_B_gt_path).convert(mode='RGB')

        image_full = image_A

        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B, image_A_gt, image_B_gt, image_full = self.transform(image_A, image_B, image_A_gt, image_B_gt, image_full)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, image_A_gt, image_B_gt, image_full, class_key[:-2], name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        images_A_gt = torch.stack(images_A_gt, dim=0)
        images_B_gt = torch.stack(images_B_gt, dim=0)
        images_full = torch.stack(images_full, dim=0)
        return images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name