# [Scientific Reports 2026] A VLM guided network coupling degradation modeling for degradation aware infrared and visible image fusion
### [Paper](https://www.nature.com/articles/s41598-026-38181-8) | [Code](https://github.com/Lmmh058/VGDCFusion) 

**A VLM guided network coupling degradation modeling for degradation aware infrared and visible image fusion (Scientific Reports 2026)**

![Framework](fig/Network.jpg)

## Prepare Your Dataset
The dataset used in this paper can be downloaded at:
[EMS](https://github.com/XunpengYi/EMS) | [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) | [MSRS](https://github.com/Linfeng-Tang/MSRS) | [M3FD](https://github.com/dlut-dimt/TarDAL) 

The images you use should be placed in:
```bash
#For Test (with degradations)
    dataset/
            {dataset}/
                      {task}/
                            Infrared/
                            Visible/
#For Train (with degradations)
    dataset/
            Text_Train/
                        {Degradation} (eg: VI_lowlight_IR_lowcontrast)/
                                                                       train/
                                                                               Infrared/
                                                                               Infrared_gt/
                                                                               Visible/
                                                                               Visible_gt/
                                                                               text_ir.txt
                                                                               text_vi.txt
```

## Pretrained Weights
Our pre-trained model is available at ```./pretrained_weights```.

## Testing
You can test the fusion performance of the model using the following command, after correctly placing the test images and the pretrained model:
```
python test_from_dataset.py
```

## Visual Results
A few qualitative examples are shown below.

![Qualitative_Comparison](fig/Qualitative_Degrade.jpg)

## Quantitative Results
Quantitative comparison examples are shown below. Higher values of all other metrics indicate better performance.

![Quantitative Comparison](fig/Quantitative_Degrade.jpg)

## Training your own model
Put your training data, and run:
```
python train_fusion.py
```
Afterwards, your model will be placed in ```./experiments```.

## Citation
If our work contributes to your research, please cite it as:
```
@article{zhao2026vlm,
  title={A VLM guided network coupling degradation modeling for degradation aware infrared and visible image fusion},
  author={Zhao, Jufeng and Zhang, Tianpei and Cui, Guangmang},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```
