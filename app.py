import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model():
    """
    Create and compile the ResNet152V2 waste classification model.
    
    Returns:
        Model: Compiled TensorFlow/Keras model
    """
    try:
        res = ResNet152V2(include_top=False, weights=None, input_shape=(224,224,3))
        x = Flatten()(res.output)
        prediction = Dense(3, activation='softmax')(x)
        model = Model(inputs=res.input, outputs=prediction)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        raise

def load_model_weights(model, weights_path='Waste_Segregation_Model.h5'):
    """
    Load pre-trained model weights.
    
    Args:
        model (Model): Keras model
        weights_path (str): Path to model weights file
    """
    try:
        model.load_weights(weights_path)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Weight loading failed: {e}")
        raise

def predict_waste_category(model, image):
    """
    Predict waste category probabilities.
    
    Args:
        model (Model): Trained Keras model
        image (PIL.Image): Input image
    
    Returns:
        dict: Prediction probabilities
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image = preprocess_input(tf.image.resize(image, [224, 224])[None, ...])
        prediction = model.predict(processed_image).tolist()[0]
        
        class_names = ['Recyclable', 'Organic', 'Non-Recyclable']
        return {class_names[i]: float(prediction[i]) for i in range(3)}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {}

def main():
    st.set_page_config(page_title="Waste Segregation", page_icon=":recycle:")
    st.title('🌍 Waste Segregation Classifier')
    st.markdown("Upload an image to classify waste type")

    try:
        model = create_model()
        load_model_weights(model)

        uploaded_file = st.file_uploader("Choose an image...", 
                                         type=["jpg", "png", "jpeg"],
                                         help="Upload a waste image for classification")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with st.spinner('Classifying...'):
                predictions = predict_waste_category(model, image)
            
            if predictions:
                st.bar_chart(predictions)
                st.write("Prediction Probabilities:")
                for category, prob in predictions.items():
                    st.write(f"{category}: {prob*100:.2f}%")
            else:
                st.error("Classification failed")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()