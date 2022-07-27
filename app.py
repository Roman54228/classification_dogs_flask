import io
import json
import torch
from torchvision import models
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
import torch.nn as nn

app = Flask(__name__)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

imagenet_class_index = json.load(open('imagenet_class_index.json'))


class_names = {0: 'Shih-Tzu',
 1: 'Rhodesian_ridgeback',
 2: 'beagle',
 3: 'English_foxhound',
 4: 'Border_terrier',
 5: 'Australian_terrier',
 6: 'golden_retriever',
 7: 'Old_English_sheepdog',
 8: 'Samoyed',
 9: 'dingo'}

# definition of model
model = models.convnext_base(pretrained=True)
model.classifier[2] = nn.Linear(in_features=1024, out_features=10, bias=True)
model.load_state_dict(torch.load('conv_next4.pth', map_location=device))
model.eval()
model.to(device)


def img_to_tensor(image_bytes):
    """Converts byte arrya to torch.tensor with transforms
    
    Args:
    -----
        img: byte
            input image as raw bytes 
    
    Returns: 
    --------
        img_tensor: torch.Tensor
            image as Tensor for using with deep learning model
    """

    # transformations for raw image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(device)


def get_prediction(image_bytes):
    """perform predictions using model defined globally
    
    Arguments:
        image_bytes:
            raw image bytes recieved via POST
    
    Returns:
        class_id: int
            id defined in imagenet_class_index.json
        class_name: str
            top predicted category 
        prob: float
            confidence score for prediction    
    """
    tensor = img_to_tensor(image_bytes=image_bytes)
    #outputs = F.softmax(model.forward(tensor))
    with torch.no_grad():
        outputs = model(tensor)
    _, predicted = torch.max(outputs.data,1)
    #prob = prob.item()
    #predicted_idx = str(y_hat.item())
    class_name = class_names[predicted.item()]
    prob = float(outputs[0][predicted.item()])
    return predicted, class_name, "{:.2f}".format(prob)


@app.route('/', methods=['GET','POST'])
def hello_world():
    """Front face of the app that defines upload and response interface
    """
    if request.method=='GET':
        return render_template('index.html',value='get')
    if request.method=='POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name, prob = get_prediction(image_bytes=img_bytes)
        
        return render_template('result.html', class_name=class_name.replace("_"," "), prob=prob)
    


@app.route('/predict', methods=['POST'])
def predict():
    """predict top category of object found in image based on imagenet pre-trained model
    This is used with requests interface in python
    Returns:
        json -- Category id and corresponding name is returned
    """
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name, prob = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
