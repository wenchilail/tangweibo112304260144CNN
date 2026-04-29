import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('best_model_final.pth', map_location=device))
model.eval()


def predict_image(image):
    img = image.convert('L')
    img = np.array(img)
    
    if np.mean(img[:10, :10]) > 128:
        img = 255 - img
    
    threshold = 128
    img = (img > threshold) * 255
    
    coords = np.argwhere(img > 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        digit = img[y_min:y_max+1, x_min:x_max+1]
        
        height, width = digit.shape
        max_dim = max(height, width)
        padding = int(max_dim * 0.3)
        new_size = max_dim + 2 * padding
        
        new_img = np.zeros((new_size, new_size), dtype=np.uint8)
        y_offset = (new_size - height) // 2
        x_offset = (new_size - width) // 2
        new_img[y_offset:y_offset+height, x_offset:x_offset+width] = digit
        img = new_img
    
    img = Image.fromarray(img.astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
    
    result = {
        'prediction': int(predicted.item()),
        'probabilities': {}
    }
    
    for i in range(10):
        result['probabilities'][str(i)] = round(float(probabilities[i].item() * 100), 2)
    
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    try:
        image = Image.open(file)
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_base64', methods=['POST'])
def predict_base64():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': '没有图片数据'}), 400
        
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("   手写数字识别系统 - Web应用启动")
    print("=" * 60)
    print("   访问地址: http://localhost:5000")
    print("   功能: 图片上传 + 手写画板")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=True)
