import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
 
# Load the VGG16 model pre-trained on the ImageNet dataset
vgg16_model = vgg16.VGG16(weights='imagenet')
 
# Read the arguments passed to the interpreter when invoking the script
image_path = sys.argv[1]
top_guesses = sys.argv[2]
 
# Load the image, resized according to the model target size
img_resized = image.load_img(image_path, target_size=(224, 224))
 
# Convert the image into an array
img = image.img_to_array(img_resized)
 
# Display the image to check that it has been correctly resized
plt.imshow(img.astype(np.uint8))
 
# Add in a dimension
img = np.expand_dims(img, axis=0) 
 
# Scale the pixel intensity values
img = preprocess_input(img) 
 
# Generate a prediction for the test image
pred_vgg = vgg16_model.predict(img)
 
# Decode and print the top 3 predictions
print('Prediction:', decode_predictions(pred_vgg, top=int(top_guesses)))