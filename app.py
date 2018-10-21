#!/usr/bin/env python
import torch
from torchvision import transforms
from transformer_net import TransformerNet
import utils
import re
import tqdm

from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

GPU=True
device = torch.device("cuda" if GPU else "cpu")

model_path = "models/rain_princess.pth"


# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/style/rain_princess', methods=['POST'])
def rain_princess():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)




# content_image_path = "images/content_images/scenery.jpg"

# 
# content_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.mul(255))
# ])
# with torch.no_grad():
#     style_model = TransformerNet()
#     state_dict = torch.load(model_path)
#     # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
#     for k in list(state_dict.keys()):
#         if re.search(r'in\d+\.running_(mean|var)$', k):
#             del state_dict[k]
#     style_model.load_state_dict(state_dict)
#     style_model.to(device)
#     for k in tqdm.tqdm(range(1000)):
#         content_image = utils.load_image(content_image_path, scale=1)
#         content_image = content_transform(content_image)
#         content_image = content_image.unsqueeze(0).to(device)        
#         output = style_model(content_image).cpu()
