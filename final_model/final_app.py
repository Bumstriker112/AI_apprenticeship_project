from Labels import Labels
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import streamlit as st
import cv2


st.title("Object detection assistance for the elderly")
menu = ['Upload image','get shape information']
choice = st.sidebar.selectbox("Menu",menu)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
if choice == 'Upload image':

	st.write("Upload image : ")
	image_file = st.file_uploader("", type = ['png','jpg','jpeg'])

	if image_file is not None:

		image = Image.open(image_file)

		
		#resize the image to a 224x224 with the same strategy as in TM2:
		#resizing the image to be at least 224x224 and then cropping from the center
		size = (224, 224)
		image = ImageOps.fit(image, size, Image.ANTIALIAS)

		#turn the image into a numpy array
		image_array = np.asarray(image)

		 

		# Normalize the image
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

		# Load the image into the array
		data[0] = normalized_image_array

		# run the inference
		prediction = model.predict(data)

		score_index_dict={}
		for score_list in prediction:
		    for score in score_list:
		        score_index = np.where(score_list==score)
		        score_index_dict[score_index[0][0]]=round(score*100,5)

		    
		# print("\n\n------------------------------------ Final Accuracy ---------------------------------\n")
		print(score_index_dict)
		max_value = max(score_index_dict.values())
		max_key = [k for k, v in score_index_dict.items() if v == max_value]
		predicted = Labels[max_key[0]]
		st.image(image)
		st.write("predicted : ", predicted)
		 