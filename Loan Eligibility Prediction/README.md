<h1 align="center">🏦 Loan Prediction System 🏦</h1>

<p align="center">
  A machine learning-based loan prediction system that predicts loan approval status based on applicant information! 🔍💡
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue"/>
  <img src="https://img.shields.io/badge/Scikit--learn-0.24-orange"/>
  <img src="https://img.shields.io/badge/Pandas-1.3.3-yellowgreen"/>
  <img src="https://img.shields.io/badge/NumPy-1.21-lightblue"/>
  <!--<img src="https://img.shields.io/github/license/basharul2002/Loan-Prediction-System"/> -->
</p>

---

## 📖 Overview

This **Loan Prediction System** leverages machine learning algorithms to predict whether a loan application will be approved or not based on various factors like applicant income, loan amount, credit history, and more. It implements both **Decision Tree** and **Naive Bayes** classifiers and provides accuracy evaluation metrics for both models.

## 💡 Features
- **Data Preprocessing**: Handling missing values, feature engineering (e.g., total income, log transformations)
- **Model Training**: Training with both Decision Tree and Naive Bayes classifiers
- **Evaluation**: Predicting on test data and calculating accuracy
- **Support for Test Data**: Test dataset handling and prediction

## 🛠️ Technologies Used

- **Python**: Programming language used for the project
- **NumPy**: For numerical operations
- **Pandas**: For data manipulation and analysis
- **Matplotlib**: For plotting boxplots and histograms
- **Scikit-learn**: For model building and evaluation

## 🗂 Dataset

- **Training Dataset**: `loan_train_data.csv`
- **Test Dataset**: `loan_test_data.csv`

The datasets contain information such as:
- **Applicant Income**
- **Coapplicant Income**
- **Loan Amount**
- **Credit History**
- **Loan Status**

---

## 🚀 How It Works

### 1️⃣ Data Preprocessing
- **Missing Values**: Handled by filling with mode/mean for categorical/numerical features.
- **Log Transformations**: Log transformation applied to `LoanAmount` and `TotalIncome` to normalize skewed distributions

### 2️⃣ Feature Engineering
- Created new features like `TotalIncome` (sum of applicant and coapplicant income) and its log transformation for better model performance

### 3️⃣ Model Training
- **Decision Tree Classifier**: Trained with entropy criterion to maximize information gain.
- **Naive Bayes Classifier**: Gaussian Naive Bayes model used for classification.

### 4️⃣ Model Evaluation
- **Accuracy** of both models evaluated on the test dataset using `accuracy_score` from **sklearn.metrics**

### 5️⃣ Test Data Prediction
- The system handles an unseen test dataset to predict loan approval statuses

---

## 📊 Data Visualization

Some key data visualizations implemented:
- **Boxplots**: For visualizing the distribution of applicant income, loan amounts
- **Histograms**: Displaying the frequency distribution of income and loan amounts

Example visualizations:

```python
# Boxplot for Applicant Income
dataset.boxplot(column='ApplicantIncome')

# Histogram for Loan Amount
dataset['LoanAmount'].hist(bins=20)
```

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Basharul2002/Data-Science-Project/tree/main/Loan%20Eligibility%20Prediction.git
```

### 2️⃣ Install Dependencies

Ensure that you have the necessary libraries installed. You can install them using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### 3️⃣ Execute the Code

Run the main script to train the model and make predictions:

```bash
python Loan Eligibility Prediction.ipynb
```

---

## 🔍 Example Output

Sample accuracy for the classifiers:

```bash
The accuracy of Decision Tree is: 0.81
The accuracy of Naive Bayes is: 0.79
```

---


## 📂 Directory Structure

```
Loan-Prediction-System/
│
├── sample_data/
│   ├── loan_train_data.csv            # Training dataset
│   ├── loan_test_data.csv             # Test dataset
│
├── Loan Eligibility Prediction.ipynb  # Main Python script
└── README.md                          # Documentation
```

<!--
---

## 📜 License

This project is licensed under the [MIT License](LICENSE)
