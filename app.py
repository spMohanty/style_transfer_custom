#!/usr/bin/env python
import torch
from torchvision import transforms
from transformer_net import TransformerNet
import utils
import re
import tqdm
import torchvision.transforms.functional as F

from io import BytesIO
from skimage.io import imsave

from flask import Flask, request, Response, send_file
import jsonpickle
import numpy as np
import cv2

from PIL import Image
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')

GPU=True
device = torch.device("cuda")

model_path = "models/rain_princess.pth"
content_image_path = "images/content_images/scenery.jpg"

content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Initialize the Flask application
app = Flask(__name__)


with torch.no_grad():
    style_model = TransformerNet()
    state_dict = torch.load(model_path)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    
    # route http posts to this method
    @app.route('/api/style/rain_princess', methods=['POST'])
    def rain_princess():
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        cv2_im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)    

        content_image = content_transform(pil_im)
        content_image = content_image.unsqueeze(0).to(device)
        output = style_model(content_image).cpu()
        output_im = transforms.ToPILImage()(output[0])
        
        bufferIO = BytesIO()
        output_im.save(bufferIO, format='JPEG')
        return Response(bufferIO.getvalue(), mimetype= 'image/jpeg')
    
    # for k in tqdm.tqdm(range(1000)):
    #     content_image = utils.load_image(content_image_path, scale=1)
    #     content_image = content_transform(content_image)
    #     content_image = content_image.unsqueeze(0).to(device)        
    #     output = style_model(content_image).cpu()



# start flask app
app.run(host="0.0.0.0", port=5001)
