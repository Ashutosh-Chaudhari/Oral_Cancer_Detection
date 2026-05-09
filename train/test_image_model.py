import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python train/test_image_model.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = "backend/models/intraoral_resnet18.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    classes = ["cancer", "normal"]
    
    print(f"Predicted class: {classes[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
