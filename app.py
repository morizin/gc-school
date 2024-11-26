# %%writefile tb-school/app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
# import efficientnet.tfkeras as efn
from PIL import Image
from tensorflow.keras.optimizers import Adam
import numpy as np

# Suppress warnings if you'd like
# import warnings
# warnings.filterwarnings('ignore')


#in_lay = Input((384,384,3))
#vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
def create_model():
    res = ResNet152V2(include_top=False, weights=None, input_shape=(224,224,3))
    # Define the sequential model
    x = Flatten()(res.output)
    prediction = Dense(3,activation='softmax')(x)
    model = Model(inputs=res.input, outputs=prediction)
    return model

model = create_model()
model.load_weights('Waste Segregation Model.h5')

# Function to preprocess the image and make predictions
def lung_defect(img):

    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Preprocessing the image to fit the model input shape
    img = preprocess_input(tf.image.resize(img, [224, 224])[None, ...])
    # img = preprocess_input(img[None, ...]
    # img_array = img_array / 255.0  # Assuming the model expects the input in this range
    # img_array = img_array.reshape((1, 224, 224, 3))  # Adjusting to the input shape

    # Make a prediction
    prediction = model.predict(img).tolist()[0]
    class_names = ['Recyclable','Organic', 'Non-Recyclable'][::-1]

    # Returning a dictionary of class names and corresponding predictions
    return {class_names[i]: float(prediction[i]) for i in range(3)}

# Streamlit user interface
st.title('Waste Segregation')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predictions = lung_defect(image)

    # Display the predictions as a bar chart
    st.bar_chart(predictions)
