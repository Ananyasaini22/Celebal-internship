#This is the streamlit app code
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern, hog
import joblib
import os

#first extracting features from the images
def extract_features(image):
    img = cv2.resize(image, (128, 128))
    features = []
    avg_color = np.mean(img, axis=(0, 1))
    features.extend(avg_color)
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp=local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist,_=np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist=lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    features.extend(lbp_hist)
    gray_resized = cv2.resize(gray, (64, 128))
    hog_features = hog(gray_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    features.extend(hog_features)
    return np.array(features),avg_color

#Now this section is for fertilizer need, basically it tells based on the greeness of the leaf
def label_fertilizer(avg_green):
    if avg_green<55:
        return "High Need"
    elif avg_green<150:
        return "Moderate Need"
    else:
        return "Low Need"
    
#loading the trained models(pickle files)
species_model = joblib.load("models/species_model.pkl")
fertilizer_model = joblib.load("models/fertilizer_model.pkl")

#the main stramlit app code
st.set_page_config(page_title="Plant Classifier", layout="wide")
st.title("Plant Species And Fertilizer Need Classifier")
uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", channels="BGR")
    with st.spinner("Analyzing"):
        features,avg_color = extract_features(image)
        species_prediction = species_model.predict([features])
        fertilizer_prediction = fertilizer_model.predict([features])
    st.subheader("Prediction Resuts")
    st.write(f"Predicted Species: {species_prediction[0]}")    
    st.write(f"Fertilizer Need: {label_fertilizer(avg_color[1])}")

 #This section is for visualizing the results, basically its a kind of heta map that visualizes the image pixels
    st.subheader("Visual Explanation")
    from skimage import exposure
    gray=cv2.cvtColor(cv2.resize(image,(128, 128)),cv2.COLOR_BGR2GRAY)
    hog_visualization,hog_image=hog(gray,orientations=9,pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2),block_norm='L2-Hys',visualize=True)
    hog_img_rescaled=exposure.rescale_intensity(hog_image,in_range=(0, 10))
    col1,col2=st.columns(2)
    with col1:
        st.markdown("Feature Visualization")
        fig1,ax1=plt.subplots()
        ax1.imshow(hog_img_rescaled, cmap='inferno')
        ax1.axis('off')
        st.pyplot(fig1)
    with col2:
        st.markdown("Feature Meaning")
        st.markdown(
            "Bright areas show strong gradients (edges & curves)\n"
            "These patterns help the model distinguish leaf textures\n"
            "Used by the model to learn structural characteristics"
        )