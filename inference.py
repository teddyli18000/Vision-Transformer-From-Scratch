import torch
import torchvision
import torchvision.transforms as transforms
from model.vit import VisionTransformer
import random

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Select a random test image
    idx = random.randint(0, len(testset)-1)
    image, label = testset[idx]

    # Initialize model
    model = VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        d_model=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1
    ).to(device)

    # Load weights
    try:
        model.load_state_dict(torch.load("vit_cifar10.pth", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'vit_cifar10.pth' not found. Please run train.py first.")
        return

    model.eval()

    # Forward pass
    image_tensor = image.unsqueeze(0).to(device) # Add batch dimension [1, C, H, W]
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    pred_class = classes[predicted.item()]
    true_class = classes[label]

    print(f"True Class:      {true_class}")
    print(f"Predicted Class: {pred_class}")

if __name__ == "__main__":
    run_inference()