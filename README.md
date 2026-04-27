# satellite-scene-classification
Multi-label satellite image classification using ResNet and EuroSAT
# 🛰️ Satellite Scene Classification using ResNet-18

**Team:** fidalgovind | Adithyan Biju | Archana T  
**Course:** Predictive Analytics  
**Dataset:** EuroSAT (16,200 satellite images, 10 classes)

## 🔴 Live Demo
[Click here to try the app](https://satellite-scene-classification-4remxga6jkwsfybr79ivjq.streamlit.app/)

## Problem Statement
Classify satellite image patches into land-use categories
using deep learning on the EuroSAT dataset.

## Dataset
- EuroSAT: 16,200 patches, 10 classes, 64x64 pixels
- Source: Sentinel-2 satellite, 13 spectral bands
- Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway,
  Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Project Pipeline
1. Problem Definition and Literature Review
2. Data Collection and Understanding
3. Data Preprocessing and Cleaning
4. Exploratory Data Analysis
5. Feature Engineering (NDVI)
6. Model Building (ResNet-18 pretrained)
7. Model Evaluation
8. Explainability (Grad-CAM)
9. Deployment (Streamlit)
10. Documentation

## Results
| Model | Accuracy | Hamming Loss |
|-------|----------|-------------|
| ResNet-18 (pretrained) | 92.89% | 0.0711 |

## Per-Class Accuracy
| Class | Accuracy |
|-------|----------|
| AnnualCrop | 96.8% |
| Forest | 91.3% |
| HerbaceousVegetation | 92.5% |
| Highway | 88.9% |
| Industrial | 98.8% |
| Pasture | 90.9% |
| PermanentCrop | 91.8% |
| Residential | 87.9% |
| River | 89.6% |
| SeaLake | 99.0% |

## Setup Instructions
pip install -r requirements.txt
streamlit run app.py

## Live Deployment
[Streamlit App](https://satellite-scene-classification-4remxga6jkwsfybr79ivjq.streamlit.app/)

## GitHub Contributors
- fidalgovind
- Adithyan Biju
- Archana T
