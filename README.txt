# 🌿 Plant Species and Fertilizer Recommendation Web App

This Streamlit web app uses traditional machine learning to classify plant species from leaf images and recommend fertilizer needs — all without deep learning or transfer learning!

Unlike typical black-box AI models, our app leverages handcrafted features such as color histograms, Local Binary Patterns (LBP), and Histogram of Oriented Gradients (HOG) for interpretable, transparent predictions.


## 🔍 Features

- 📸 Upload a leaf image to receive:
  - ✅ Predicted plant species (Tomato, Potato, Bell Pepper)
  - 🌾 Fertilizer need level (High, Moderate, Low)
- 📊 Get model interpretability via:
  - HOG visualization of important leaf textures
  - Color-based fertilizer logic based on leaf greenness
- 🔬 Built using:
  - Support Vector Machines (SVM)
  - Random Forest Classifiers
- ❌ No deep learning or transfer learning — purely classical machine learning!



## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/plant-classifier-app.git
cd plant-classifier-app

Install dependencies:
pip install -r requirements.txt

Run the App
streamlit run app_plant.py

Dataset Used
This app uses a subset of the PlantVillage dataset with custom-labeled images:
Species: Potato, Tomato, Bell Pepper
Folders include both healthy and diseased leaf samples.
Images were manually curated and balanced for class distribution.


Model Performance
| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------- | --------- | ------ | -------- |
| SVM (Species) | 97.8%    | 97.3%     | 100%   | 98.6%    |
| Random Forest | 96.5%    | 95.8%     | 98.5%  | 97.2%    |

