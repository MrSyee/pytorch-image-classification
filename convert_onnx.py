import torch
import torch.nn as nn
import torchvision.models as models

# Constuct model
model = models.__dict__["resnet152"](pretrained=True)
model.fc = nn.Linear(2048, 2)
model = torch.nn.DataParallel(model)
print(type(model))
print(model)

# Load model
ckpt_path = "./models/model_best.pth"
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["state_dict"])
# unwrap from DataParallel
model = model.module.cpu().eval()

# Convert onnx
dummy_input = torch.randn(1, 3, 224, 224)
onnx_file_name = "./models/resnet152.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_name,
    input_names=["input"],
    output_names=["output"],
)
