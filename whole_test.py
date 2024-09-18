import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Define the path to the saved model and test images
model_path = 'model.pth'
test_folder = 'dataset/test/test'
output_folder = 'path_to_save_results'  # Optional: Save the results here

# Define the class labels (update this based on your specific classes)
class_labels = ['Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)_Powdery_mildew',
        'Cherry_(including_sour)_healthy',
        'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)_Common_rust',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)_healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy',
        'blast',
        'blight',
        'tungro']

# Load the model
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet101(pretrained=False, num_classes=len(class_labels))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the transformations for the test images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Test the images
for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    if os.path.isfile(image_path):
        predicted_class = predict_image(image_path)
        predicted_label = class_labels[predicted_class]
        
        # Save or print the result
        result = f'{image_name}: {predicted_label}'
        print(result)
        
        # Optionally save the result to a file
        with open(os.path.join(output_folder, 'results.txt'), 'a') as f:
            f.write(result + '\n')

print('Testing complete!')
