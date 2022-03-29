from pathlib import Path
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


import torch
import torch.nn as nn
import torchvision.models as models

# Constuct model
model = models.__dict__["resnet152"](pretrained=True)
model.fc = nn.Linear(2048, 2)
model = torch.nn.DataParallel(model)
# print(type(model))
# print(model)

# Load model
ckpt_path = "./models/model_best.pth"
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["state_dict"])
# unwrap from DataParallel
model = model.module.cpu().eval()

# Inference
## preprocessing
trasnforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ]
)

## load image
### ["./data/cat.0.jpg", "./data/cat.1.jpg", "./data/cat.2.jpg", "./data/dog.0.jpg", "./data/dog.1.jpg"]
data_path = Path("./data/cat.1.jpg")
image = Image.open(data_path)
image_tensor = trasnforms(image)
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.type(), image_tensor.size())

# image_var = torch.autograd.Variable(images)
y_pred = model(image_tensor)
# get the index of the max log-probability
smax = nn.Softmax(dim=1)
smax_out = smax(y_pred)[0]
cat_prob = smax_out.data[0]
dog_prob = smax_out.data[1]

print(f"[torch model] cat: {cat_prob}\t dog: {dog_prob}")


# Load onnx model
import onnxruntime
import numpy as np

onnx_model = onnxruntime.InferenceSession("./models/resnet152.onnx")
input_name = onnx_model.get_inputs()[0].name
output_name = onnx_model.get_outputs()[0].name

# onnx result
y_pred = onnx_model.run(None, {input_name: image_tensor.cpu().numpy()})[0]
y_pred = torch.from_numpy(y_pred)
smax = nn.Softmax(dim=1)
smax_out = smax(torch.from_numpy(np.array(y_pred)))[0]
cat_prob = smax_out.data[0]
dog_prob = smax_out.data[1]

print(f"[onnx model] cat: {cat_prob}\t dog: {dog_prob}")
