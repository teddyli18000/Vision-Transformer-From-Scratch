import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from model.vit import VisionTransformer


def predict(image_path, model_path="vit_cifar10.pth"):
    # 1. Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' does not exist.")
        return

    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Define the exact same transformations used during training/testing
    # Note: We MUST resize the image to 32x32 because our ViT expects 32x32 inputs
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 4. Load and preprocess the image
    try:
        # Convert to RGB to ensure it has 3 channels (ignores alpha channels in PNGs)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension: [1, 3, 32, 32]
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return

    # 5. Initialize the model with the exact same architecture parameters
    model = VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        d_model=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1
    ).to(device)

    # 6. Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights '{model_path}' not found. Please run train.py first.")
        return

    model.eval()

    # CIFAR-10 class labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 7. Perform inference
    with torch.no_grad():
        output = model(image_tensor)

        # Get the predicted class index
        _, predicted_idx = torch.max(output, 1)

        # Optional: Get confidence scores using Softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence = probabilities[predicted_idx.item()].item() * 100

    # 8. Output the result
    pred_class = classes[predicted_idx.item()]

    print("-" * 30)
    print(f"Image: {image_path}")
    print(f"Predicted Content: {pred_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    print("Type 'exit' or 'quit' to terminate.")

    # 默认权重路径
    default_model = "vit_cifar10.pth"

    while True:
        path = input("\nInput the path (eg. my_cat.jpg): ").strip()

        # 退出逻辑
        if path.lower() in ['exit', 'quit', '']:
            print("Terminated")
            break

        # 执行预测
        predict(path, default_model)