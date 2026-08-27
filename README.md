# Electric Vehicle Range Prediction Using Machine Learning

This project predicts the estimated driving range of an Electric Vehicle using Machine Learning based on battery capacity, vehicle weight, average speed, and temperature.

## Live Demo

🚀 [EV Range Predictor](https://electric-vehicle-range-prediction-using-ml.streamlit.app/)

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

## Dataset Features

The dataset contains the following features:

- Battery Capacity (kWh)
- Vehicle Weight (kg)
- Average Speed (km/h)
- Temperature (°C)
- Range (km) – Target variable

## Machine Learning Model

A **Linear Regression** model is used to predict the EV driving range.

The model takes the following parameters as input:

- Battery Capacity
- Vehicle Weight
- Average Speed
- Temperature

and predicts the estimated driving range in kilometers.

## Project Files

- `EV_range_prediction.py` – Machine learning model training and evaluation
- `app.py` – Streamlit application for interactive predictions
- `ev_dataset.csv` – Dataset used for training and testing
- `ev_range_model.pkl` – Saved trained model
- `train_model.py` – Script for training and saving the model
- `requirements.txt` – Required Python libraries
- `README.md` – Project documentation

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Bhavani-adabala24/Electric-Vehicle-Range-Prediction-Using-Machine-Learning.git
