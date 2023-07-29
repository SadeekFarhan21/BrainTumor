import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

INPUT_SIZE = 64

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('/home/farhan/Documents/BrainTumor/pred/pred5.jpg')
image = Image.fromarray(image)
image = image.resize((INPUT_SIZE, INPUT_SIZE))
image = np.array(image)
input_image = np.expand_dims(image, axis = 0)

result = model.predict(input_image)
print(result)