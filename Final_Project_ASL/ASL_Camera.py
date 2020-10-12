from tensorflow.keras.models import model_from_json

import cv2
import numpy as np

json_file = open("model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("model.h5")

labels = list("ABC")

def image_predict(image):
    return labels[np.argmax(model.predict(image))]