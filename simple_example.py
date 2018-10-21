#!/usr/bin/env python
import torch
from torchvision import transforms
from transformer_net import TransformerNet
import utils
import re
import tqdm

GPU=True
device = torch.device("cuda" if GPU else "cpu")

model_path = "models/rain_princess.pth"
content_image_path = "images/content_images/scenery.jpg"


content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
with torch.no_grad():
    style_model = TransformerNet()
    state_dict = torch.load(model_path)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    for k in tqdm.tqdm(range(1000)):
        content_image = utils.load_image(content_image_path, scale=1)
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)        
        output = style_model(content_image).cpu()
