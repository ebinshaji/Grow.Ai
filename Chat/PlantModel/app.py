from flask import  Flask, request, render_template, jsonify
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

saved_state_dict = torch.load("plant1_with_class_names.pth", map_location=torch.device('cpu'))

# Extract class names (copy from existing code)
if 'class_names' in saved_state_dict:
    class_names = saved_state_dict['class_names']
else:
    print("Class names not found in the state dictionary.")

# Load the trained model (copy from existing code)
model = MRW_CNN(num_classes=len(class_names))
model.load_state_dict(saved_state_dict['state_dict'])
model.eval()

# Define image transformation (copy from existing code)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        image_file = request.files['image']
        if image_file:
            # Preprocess the image
            image = Image.open(image_file).convert("RGB")
            image = transform(image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()

            # Get predicted class name
            if class_names:
                    predicted_class_name = class_names[predicted_class]
                    # Split the class name into words
                    class_words = predicted_class_name.split()
                    # Remove duplicates
                    unique_words = list(set(class_words))
                    if len(unique_words) >= 2:
                            unique_words.reverse()
                    # Join the unique words back into a single string
                    unique_class_name = ' '.join(unique_words)
                    return unique_class_name
            else:
                return jsonify({'error': 'Class names not available.'})
        else:
            return jsonify({'error': 'No image uploaded.'})

if __name__ == '__main__':
    app.run(debug=True)