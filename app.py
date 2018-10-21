#!/usr/bin/env python

import onnx
import onnx_caffe2.backend


model_path = "models/rain_princess.onnx"
content_image = "images/content_images/scenery.jpg"

model = onnx.load(model_path)

prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA')
inp = {model.graph.input[0].name: content_image.numpy()}
