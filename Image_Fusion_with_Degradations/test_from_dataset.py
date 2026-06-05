import os
import numpy as np
from PIL import Image
import cv2
import clip
import torch
from torchvision.transforms import functional as F
from model.VGDCFusion_model import VGDCFusion as create_model
import argparse
import torch.nn.functional as F_nn 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def pad_to_multiple(img_tensor, multiple=16):
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    img_padded = F_nn.pad(img_tensor, padding, mode='reflect')
    return img_padded, (h, w)

def crop_to_original(img_tensor, original_size):
    h, w = original_size
    return img_tensor[:, :, :h, :w]

def get_input_texts(task):
    text_map = {
        "LowlightContrast": (
            "This task involves the fusion of infrared and visible light images, with the infrared images suffering from low contrast degradation.",
            "The task at hand is the fusion of infrared and visible light images, where visible images are affected by low light degradation."
        ),
        "LowlightNoise": (
            "Infrared-visible light image fusion is our focus, and it's crucial to address the noise degradation in the infrared images.",
            "The task at hand is the fusion of infrared and visible light images, where visible images are affected by low light degradation."
        ),
        "OverexposureContrast": (
            "This task involves the fusion of infrared and visible light images, with the infrared images suffering from low contrast degradation.",
            "We are tackling the infrared-visible light image fusion task, focusing on correcting the overexposure degradation in visible images."
        ),
        "OverexposureNoise": (
            "Infrared-visible light image fusion is our focus, and it's crucial to address the noise degradation in the infrared images.",
            "We are tackling the infrared-visible light image fusion task, focusing on correcting the overexposure degradation in visible images."
        ),
        "none": ("", "")
    }
    return text_map.get(task, ("", ""))

def main(args):
    # Automatically construct paths
    if args.task == "none":
        dataset_path = os.path.join("./dataset", args.dataset)
        save_path = os.path.join(f"./results", args.dataset)
    else:
        dataset_path = os.path.join("./dataset", args.dataset, args.task)
        save_path = os.path.join(f"./results", args.dataset, args.task)

    weights_path = os.path.join("./pretrained_weights", "checkpoint.pth")
    text_line_ir, text_line_vi = get_input_texts(args.task)
    print('text_ir:{}, text_vi:{}'.format(text_line_ir, text_line_vi))

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']

    visible_root = os.path.join(dataset_path, "Visible")
    infrared_root = os.path.join(dataset_path, "Infrared")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model = create_model(model_clip).to(device)

        # model_weight_path = args.weights_path
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = vi_path.replace("\\", "/").split("/")[-1]
        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        vi_tensor = F.to_tensor(vi).unsqueeze(0).to(device)
        ir_tensor = F.to_tensor(ir).unsqueeze(0).to(device)

        vi_padded, original_size = pad_to_multiple(vi_tensor)
        ir_padded, _ = pad_to_multiple(ir_tensor)
        print("Input visible image size:", vi_padded.shape)
        print("Input infrared image size:", ir_padded.shape)

        with torch.no_grad():
            text_ir = clip.tokenize(text_line_ir).to(device)
            text_vi = clip.tokenize(text_line_vi).to(device)

            output = model(vi_padded, ir_padded, text_ir, text_vi)
            output_cropped = crop_to_original(output, original_size)
            fused_img_Y = tensor2numpy(output_cropped)

            save_pic(fused_img_Y, save_path, img_name)

        print("Save the {}".format(img_name))
    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).detach().cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])
    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index) 
    cv2.imwrite(save_path, outputpic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Text_Test', help='Dataset name, e.g., "Text_Test"')
    parser.add_argument('--task', type=str, default='LowlightContrast', choices=[
        'LowlightContrast', 'LowlightNoise', 'OverexposureContrast', 'OverexposureNoise', 'none'
    ], help='Degradation type or "none"')
    parser.add_argument('--experiment', type=str, default='', help='Experiment folder name')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)