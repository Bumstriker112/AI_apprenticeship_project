from Labels import Labels
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import streamlit as st
import cv2

st.title("ElderHelp.ai : Object detection assistance for the elderly")
menu = ['Upload image','About us']
choice = st.sidebar.selectbox("Menu",menu)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if choice == 'Upload image':
	st.write("Upload image : ")
	image_file = st.file_uploader("", type = ['jpg','jpeg'])

	if image_file is not None:
		image = Image.open(image_file)
		size = (224, 224)
		image = ImageOps.fit(image, size, Image.ANTIALIAS)
		image_array = np.asarray(image)
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
		# Load the image into the array
		data[0] = normalized_image_array
		# run the inference
		prediction = model.predict(data)

		#obtain max score
		score_index_dict={}
		for score_list in prediction:
		    for score in score_list:
		        score_index = np.where(score_list==score)
		        score_index_dict[score_index[0][0]]=round(score*100,5)

		print(score_index_dict)
		max_value = max(score_index_dict.values())
		max_key = [k for k, v in score_index_dict.items() if v == max_value]
		predicted = Labels[max_key[0]]
		
		min_req_acc = st.slider("Select minimum required confidence score",min_value = 0, max_value = 100, value=50)

		if(max_value >= min_req_acc):
			st.image(image)
			st.write("predicted : ", predicted)
			see_accuracy = st.checkbox("See confidence score")
			if see_accuracy:
				st.write("confindence score : " , max_value , "%")
		else:
			st.write("No object was able to be predicted on that minimum required confidence score")

else:
	st.title("About us")
	st.write("ElderHelp.ai is an project made by Team 'AI Overlords' as a part of CBSE-Intel apprenticeship program.")
	st.write("Due to old age, short term memory lapses occur slowing our thinking processes. This is accompanied by difficulties in identifying everyday objects including their shape and size")
	st.write("The team has tried to create an AI solution to tackle this problem and create a social impact")
	st.write("The final model trained contains 25 classes to predict upon :  basket bin, bed, bench, cabinet, call bell, cane stick, chair, door, electric socket, fan, fire extinguisher, handrail, human being, rack, refrigerator, shower, sink, sofa, table, television, toilet seat, walker, wardrobe, water dispenser, wheelchair")
	st.write("The team is always open for feedback and constructive criticism. Contact us at : ")
	st.write("Rishit Malpani : cool.rishit@gmail.com")
	st.write("Pragyan Srivastava : prag12241@gmail.com")
