# Vision-Transformer-From-Scratch
Vision Transformer Implementation From Scratch

A clean, stable implementation of a Vision Transformer (ViT) in PyTorch, targeting the CIFAR-10 dataset.

## Features
- **No `nn.TransformerEncoder`**: Multi-Head Attention and Transformer Blocks are implemented from scratch.
- **Stable Training**: Uses Pre-LayerNorm, truncated normal weight initialization, and gradient clipping.
- **Lightweight**: Configured to easily train on a single 8GB VRAM GPU (e.g., RTX 5060Laptop).
- **Expected Performance**: Reaches ~65% - 70% validation accuracy on CIFAR-10 in just 20 epochs.


## Current version
* **v2.0** -feat: add predict.py to support interactive input 



## Setup
Ensure you have PyTorch and Torchvision installed:
```
pip install torch torchvision
```
## Usage
### 1. Train the Model
To download the dataset and start training:
```
python train.py
```
This will train for 20 epochs and save the weights to vit_cifar10.pth.
### 2. Option 1: Run Inference in terminal
To test the model on a random image from the test set:
```python inference.py```
### Option 2 **(Recommend)**: Predict on custom images (Interactive Mode)
To test the model on your own local image files, use the interactive prediction script. This allows you to process multiple images without restarting the program.
```python predict.py```

Example Session:
```
Type 'exit' or 'quit' to stop.

Input the path (eg. my_cat.jpg): images/test_sample.jpg
------------------------------
Image: images/test_sample.jpg
Predicted Content: DOG
Confidence: 94.21%
------------------------------

Input the path (eg. my_cat.jpg): exit
Terminated
```


## Historical version

### For major version update history, see ["version-log.md"](./version-log.md)

* **v2.0** -feat: add predict.py to support interactive input 


* **v1.9** -docs: standardize document
* **v1.0** Architecture