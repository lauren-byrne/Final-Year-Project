import cv2
import numpy as np
import requests
import io
import json

img = cv2.imread('test.png')
height, width, _ = img.shape

url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", img, [1, 90])
file_bytes = io.BytesIO(compressedimage)

result = requests.post(url_api,
              files = {"test.png": file_bytes},
              data = {"apikey": "13a191749988957"})

result = result.content.decode()
result = json.loads(result)

print(result)
text_detected = result.get("ParsedResults")[0].get("ParsedText")
print('text: ', text_detected)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()