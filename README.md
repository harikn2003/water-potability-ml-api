# 💧 Water Potability Prediction API (Ensemble ML)

## 🚀 Overview

This project focuses on building and deploying a machine learning model to predict whether water is safe for drinking.
The trained model is exposed as a REST API using FastAPI and containerized using Docker for easy deployment and scalability.

The project demonstrates end-to-end ML engineering, including model training, deployment, API integration, and containerization.

---

## 🧠 Model Details

* Problem Type: Binary Classification (Potable vs Not Potable)
* Algorithms Used:

  * Random Forest Classifier
  * Gradient Boosting Classifier
  * Logistic Regression
* Ensemble Method: Soft Voting Classifier
* Accuracy: ~70–80%
* ROC-AUC Score: ~0.7–0.85

---

## ⚙️ Tech Stack

* Python
* FastAPI (API Development)
* Scikit-learn (Machine Learning)
* Pandas & NumPy (Data Processing)
* Pydantic (Data Validation)
* Docker & Docker Compose (Containerization)
* Git & GitHub (Version Control)

---

## 🔧 Features

* REST API for real-time ML predictions
* Ensemble learning for improved model performance
* Input validation using Pydantic
* Logging for monitoring predictions and debugging
* Health check endpoint for service monitoring
* Fully Dockerized for consistent deployment

---

## 📂 Project Structure

```
water-potability-ml-api/
│
├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model & scaler loader
│   ├── schema.py        # Input validation schema
│
├── model/
│   ├── model.pkl        # Trained ML model
│   ├── scaler.pkl       # Preprocessing scaler
│
├── train.py             # Model training script
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
├── README.md            # Project documentation
├── .gitignore
```

---

## ▶️ Run Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python train.py
```

### 3. Start FastAPI server

```
uvicorn app.main:app --reload
```

### 4. Open in browser

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

### 1. Build and run using Docker Compose

```
docker-compose up --build
```

### 2. Access API

```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### 🔹 Home

```
GET /
```

### 🔹 Health Check

```
GET /health
```

### 🔹 Prediction

```
POST /predict
```

---

## 📥 Sample Request

```json
{
  "ph": 7,
  "Hardness": 200,
  "Solids": 15000,
  "Chloramines": 7,
  "Sulfate": 300,
  "Conductivity": 400,
  "Organic_carbon": 10,
  "Trihalomethanes": 80,
  "Turbidity": 4
}
```

---

## 📤 Sample Response

```json
{
  "prediction": 1,
  "probability": 0.82,
  "result": "Safe"
}
```

---

## 🧠 How It Works

1. Data is preprocessed (handling missing values + scaling)
2. Multiple models are trained
3. Ensemble model combines predictions using soft voting
4. Model is saved using pickle
5. FastAPI loads model and serves predictions via REST API
6. Docker ensures consistent deployment across environments

---

## 🎯 Key Highlights

* End-to-end ML pipeline from training to deployment
* Production-style API using FastAPI
* Dockerized service for scalability
* Real-time prediction capability
* Clean and modular code structure

---


## 👨‍💻 Author

Harikrishnan N

* GitHub: https://github.com/harikn2003

---
