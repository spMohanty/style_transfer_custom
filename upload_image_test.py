import requests
import json
import cv2

addr = 'http://localhost:5001'
test_url = addr + '/api/style/rain_princess'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('images/content_images/scenery.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))
