import streamlit as st 
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

CLASS_TYPE = {0:'Covid-19', 1:'Normal'}
st.set_page_config(page_title='Covid-19 X Ray Classifier',initial_sidebar_state="expanded")
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('Chest X-Ray Image Classifier to Detect Covid-19')

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_trained_model():
    model = load_model('Covid_model_5.h5')
    return model

def get_img_array(img_path):
    size=(224,224)
    image = ImageOps.fit(img_path,size)
    img=np.asarray(image)
    img=np.expand_dims(img, axis=0)
    return img

def prediction(processed_image):
    pred=CLASS_TYPE[np.argmax(model.predict(img))]
    return pred
    
img_data = st.sidebar.file_uploader(label='', type=['png','jpg','jpeg'])
	
try:
	if img_data is None:
		st.subheader("Upload a Chest X-ray Image for Classification")
		
	else:
		model=load_trained_model()
		
		uploaded_img=Image.open(img_data)
		st.image(uploaded_img, use_column_width=True)
		
		img=get_img_array(uploaded_img)
		result = prediction(img)
		st.subheader('Results:')
		st.write(f"The given X-Ray image is of type : {result}")
		st.write()
		st.write(f"The chances of patient having Covid-19 is : {round(model.predict(img)[0][0]*100,3)} %")
		st.write()
		st.write(f"The chances of patient being Normal is : {round(model.predict(img)[0][1]*100,3) }%")
	
except:
	st.write("Sorry, there was an error while processing the results. The image might be of an invalid format.")    
  
 
    
    
    
    
    