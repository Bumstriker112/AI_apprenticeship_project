from Labels import Labels
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import streamlit as st
import cv2

classes = {
	'trash bin' : 'cylindrical',
	'bed' : 'rectangular',
	'bench' : 'rectangular',
	'cabinet' : 'rectangular',
	'call bell' : 'rectangular',
	'cane stick' : 'thin cylinder',
	'chair' : 'rectangular with 4 cylindrical feet',
	'door' : 'rectangular',
	'electric socket' : 'rectangular',
	'fan' : 'circular',
	'fire extinguisher' : 'cylndrical',
	'handrail' : 'cylindrical with curved edges',
	'human being' : 'complex shape, contains many',
	'rack' : 'rectangular',
	'refrigerator' : 'rectangular',
	'shower' : 'curver cylinder with circular head',
	'sink' : 'rectangular/circular',
	'sofa' : 'rectangular',
	'table' : 'rectangular/circular',
	'television' : 'rectangular',
	'toilet seat' : 'oval/circular/rectangle',
	'walker' : 'long cylindrical metal legs',
	'wardrobe' : 'rectangular',
	'water dispenser' : 'rectangular with cylidrical water container',
	'wheelchair' : 'seating area with circular wheels'
}
st.title("ElderHelp.ai : Object detection assistance for the elderly")
menu = ['Upload image','get shape information','About us']
choice = st.sidebar.selectbox("Menu",menu)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if choice == 'Upload image':
	st.write("Upload image : ")
	image_file = st.file_uploader("", type = ['png','jpg','jpeg'])

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
		
		min_req_acc = st.slider("Select minimum required predicted accuracy",min_value = 0, max_value = 100, value=50)

		if(max_value >= min_req_acc):
			st.image(image)
			st.write("predicted : ", predicted)
			see_accuracy = st.checkbox("See accuracy")
			if see_accuracy:
				st.write("Accuracy : " , max_value , "%")
		else:
			st.write("No object was able to be predicted on that minimum required accuracy")

elif choice == 'get shape information' :
	st.write("Enter the object whose shape details are required")
	required_object = st.text_input("")
	if required_object in classes:
		st.write("Shape of : ", required_object, " is : ", classes[required_object])

else:
	st.title("About us")
	st.write("ElderHelp.ai is an project made by Team 'AI Overlords' as a part of CBSE-Intel apprenticeship program.")
	st.write("Due to old age, short term memory lapses occur slowing our thinking processes. This is accompanied by difficulties in identifying everyday objects including their shape and size")
	st.write("The team has tried to create an AI solution to tackle this problem and create a social impact")
	st.write("The final model trained contains 25 classes to predict upon :  basket bin, bed, bench, cabinet, call bell, cane stick, chair, door, electric socket, fan, fire extinguisher, handrail, human being, rack, refrigerator, shower, sink, sofa, table, television, toilet seat, walker, wardrobe, water dispenser, wheelchair")
	st.write("The team is always open for feedback and constructive criticism. Contact us at : ")
	st.write("Rishit Malpani : cool.rishit@gmail.com")
	st.write("Pragyan Srivastava : prag12241@gmail.com")