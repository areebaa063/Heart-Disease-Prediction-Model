# ❤️ Heart Disease Prediction Model

A Machine Learning project that predicts the likelihood of heart disease using a Support Vector Classifier (SVC). The project uses the **UCI Heart Disease (Cleveland) dataset** and includes a Streamlit web application for interactive predictions.

**Created by Abdul Muqeet**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)

---

## 📌 Project Overview

Heart disease is a major global health concern. This project demonstrates how machine learning can be applied to a healthcare-related classification problem.

The model uses four selected features to estimate whether heart disease is likely based on the user's inputs. A Streamlit application provides a simple interface for entering these features and viewing the prediction.

This project was developed for a Machine Learning assignment to practice data preprocessing, feature encoding, model training, prediction, and deployment.

## ✨ Features

* **Machine Learning Classification:** Uses a Support Vector Classifier (SVC).
* **Interactive Web App:** Enter patient-related feature values through Streamlit.
* **Data Preprocessing:** Includes encoded datasets and prepared training/testing data.
* **Categorical Encoding:** Converts categorical inputs into numerical values consistent with the trained model.
* **Real-Time Predictions:** Generates a prediction when the user submits the form.
* **Prediction Transparency:** Allows users to inspect the encoded feature vector sent to the model.
* **GitHub Integration:** Includes a link to the creator's GitHub profile.

## 🧠 Machine Learning Model

The project uses a **Support Vector Classifier (SVC)**, a supervised machine learning algorithm commonly used for classification tasks.

The model was trained using four selected features:

| Feature                 | Description                           |
| ----------------------- | ------------------------------------- |
| Sex                     | Sex category                          |
| Chest Pain Type         | Type of chest pain experienced        |
| Exercise-Induced Angina | Whether angina occurs during exercise |
| Maximum Heart Rate      | Grouped maximum heart rate achieved   |

The application encodes these inputs and passes them to the trained model to generate a prediction.

### Prediction Output

The application displays one of two outcomes:

* **Heart Disease Likely:** The model predicts the positive class.
* **Heart Disease Not Likely:** The model predicts the negative class.

These results represent the model's classification, not a medical diagnosis.

## 🗂️ Project Structure

```text
Heart-Disease-Prediction-Model/
│
├── Heart_Disease_Prediction_1.ipynb  # Jupyter Notebook and ML workflow
├── app.py                            # Streamlit web application
├── heart_disease_svc_model.pkl       # Trained SVC model
├── requirements.txt                  # Python dependencies
│
├── heart-data.csv                    # Dataset
├── heart-disease.names               # Dataset documentation
├── processed.cleveland.data          # Original Cleveland dataset file
│
├── heart-data-encoded.csv            # Encoded dataset
├── heart-data-encoded-output.csv     # Encoded dataset output
├── training-data-encoded.csv         # Encoded training data
├── testing-data-encoded.csv          # Encoded testing data
├── model-predictions.csv              # Model prediction results
│
└── .gitignore                        # Files excluded from Git
```

## 🛠️ Technologies Used

* **Python** — Programming language
* **Pandas** — Data manipulation and preprocessing
* **Scikit-learn** — Label encoding and machine learning
* **NumPy** — Numerical computing through the Python ecosystem
* **Jupyter Notebook** — Model development and experimentation
* **Streamlit** — Interactive web application
* **Git and GitHub** — Version control and project hosting

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/amuqeet6041/Heart-Disease-Prediction-Model.git
```

### 2. Navigate to the Project Directory

```bash
cd Heart-Disease-Prediction-Model
```

### 3. Create a Virtual Environment (Recommended)

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it doesn't, visit:

`http://localhost:8501`

## 🚀 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with your GitHub account.
3. Create a new application.
4. Select the repository `amuqeet6041/Heart-Disease-Prediction-Model`.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

Once deployment completes, Streamlit will provide a public URL for the application.

## 📊 Dataset

This project uses the **UCI Heart Disease dataset**, specifically the Cleveland dataset.

The dataset is commonly used for educational machine learning experiments involving heart disease classification.

Dataset source: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

The project repository contains dataset files and processed versions used during development.

## ⚠️ Disclaimer

**This project is for educational purposes only.**

The model uses only four selected features and a limited dataset. Its predictions must not be used to diagnose, rule out, or treat heart disease.

The application is not a substitute for professional medical advice, clinical evaluation, or diagnostic testing. Anyone experiencing health concerns should consult a qualified healthcare professional.

## 👨‍💻 Author

**Abdul Muqeet**

* GitHub: [@amuqeet6041](https://github.com/amuqeet6041)
* Project Repository: [Heart Disease Prediction Model](https://github.com/amuqeet6041/Heart-Disease-Prediction-Model)

---

⭐ If you find this project useful for learning machine learning, consider giving the repository a star!
