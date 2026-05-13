# 🧬 MediPredict AI — Diabetes Risk Analysis System

> An AI-powered web application that predicts diabetes risk using Machine Learning.
> Developed by **Vishal Jha** | Final Year Project | Computer Science | 2026

---

## 🚀 Live Demo

Run the app locally using the steps below.

---

## 📌 About The Project

**MediPredict AI** is an advanced diabetes prediction system built with Python and Streamlit. It takes 8 medical parameters as input and uses a trained Machine Learning model to predict whether a patient is at risk of diabetes or not — with confidence scores.

---

## ✨ Features

- 🤖 Trained on 3 ML models — Logistic Regression, Random Forest, and SVM
- 📊 81% prediction accuracy
- ⚡ Real-time prediction with confidence percentage
- 🎨 Beautiful futuristic UI built with Streamlit
- 📈 Visual charts — Heatmap, Confusion Matrix, Model Comparison
- 🔢 8 medical parameters analyzed

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Scikit-learn | Machine Learning models |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Data visualization |
| Pickle | Model saving/loading |

---

## 📂 Project Structure

```
disease-prediction/
│
├── app.py                  # Main Streamlit web app
├── train_model.py          # ML model training script
├── model.pkl               # Trained ML model
├── scaler.pkl              # Data scaler
├── requirements.txt        # Python dependencies
├── Run_Project.bat         # One-click run script (Windows)
│
├── dataset/
│   └── diabetes.csv        # Pima Indians Diabetes Dataset
│
├── confusion_matrix.png    # Model confusion matrix
├── heatmap.png             # Feature correlation heatmap
├── model_comparison.png    # Accuracy comparison chart
└── outcome_distribution.png # Diabetes outcome chart
```

---

## ⚙️ How To Run

### 1. Clone the repository
```bash
git clone https://github.com/Vishaljha111/disease-prediction.git
cd disease-prediction
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
```

### 3. Run the app
```bash
streamlit run app.py
```

Or simply double-click **Run_Project.bat** on Windows!

---

## 🔢 Input Parameters

| Parameter | Description | Unit |
|---|---|---|
| Pregnancies | Number of pregnancies | count |
| Glucose Level | Blood glucose concentration | mg/dL |
| Blood Pressure | Diastolic blood pressure | mm Hg |
| Skin Thickness | Triceps skin fold thickness | mm |
| Insulin Level | 2-hour serum insulin | μU/mL |
| BMI | Body Mass Index | kg/m² |
| Diabetes Pedigree | Diabetes family history score | score |
| Age | Age of the patient | years |

---

## 🤖 ML Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | ~78% |
| Random Forest | ~81% |
| SVM | ~80% |

The best performing model is automatically selected and saved.

---

## ⚠️ Disclaimer

> This application is for **educational purposes only** and is not a substitute for professional medical advice. Always consult a certified physician for medical diagnosis.

---

## 👨‍💻 Developer

**Vishal Jha**
Final Year | Computer Science | AI / Machine Learning | 2026

[![GitHub](https://img.shields.io/badge/GitHub-Vishaljha111-black?style=flat&logo=github)](https://github.com/Vishaljha111)
