import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

class MRW_CNN(nn.Module):
    def __init__(self, num_classes):
        super(MRW_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Load the saved state dictionary
# saved_state_dict = torch.load("plant1_with_class_names.pth")
# Load the saved state dictionary
saved_state_dict = torch.load("plant1_with_class_names.pth", map_location=torch.device('cpu'))

# Print out the keys of the saved state dictionary
# print("Keys in the saved state dictionary:", saved_state_dict.keys())

# Extract class names from the saved state dictionary if available
if 'class_names' in saved_state_dict:
    class_names = saved_state_dict['class_names']
   # print("Class names:", class_names)
#else:
   # print("Class names not found in the state dictionary.")

# Load the trained model
model = MRW_CNN(num_classes=len(class_names))  # Initialize your model with the number of classes
# Identify the correct key for loading the model state dictionary
model_state_dict_key = 'state_dict'  # Modify this according to your saved state dictionary
model.load_state_dict(saved_state_dict[model_state_dict_key])  # Load trained weights
model.eval()  # Set the model to evaluation mode

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = "Apple_Apple_scab1009.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Perform the prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

# Get the predicted class name
if class_names:
    predicted_class_name = class_names[predicted_class]
    print("Predicted class:", predicted_class_name)
else:
    print("Class names not available.")
