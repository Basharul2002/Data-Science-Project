# Loan Eligibility Prediction

This project aims to predict loan eligibility based on various features such as applicant's income, loan amount, credit history, etc. The project involves data cleaning, preprocessing, model building, and evaluation using classification algorithms like Decision Tree and Naive Bayes.

## Project Structure

The main file used in this project is `Loan Eligibility Prediction.ipynb`. The dataset used for training and testing can be found in the `./sample_data` folder.

### Dataset

The dataset contains information about applicants' income, loan amount, credit history, and other related attributes. 

- **Training Data**: `loan_train_data.csv`
- **Testing Data**: `loan_test_data.csv`

### Features of the Dataset

- **ApplicantIncome**: The income of the loan applicant.
- **CoapplicantIncome**: The income of the co-applicant (if any).
- **LoanAmount**: The loan amount requested.
- **Loan_Amount_Term**: Term of the loan in months.
- **Credit_History**: Credit history (1 means good credit, 0 means bad credit).
- **Loan_Status**: Whether the loan was approved or not (Y/N).
- Other features include `Gender`, `Married`, `Education`, `Self_Employed`, etc.

### Data Preprocessing

The following preprocessing steps were performed:

- Handled missing values by filling them with either the mode or mean of the respective columns.
- Created new features such as `TotalIncome` and `TotalIncome_log`.
- Applied log transformations on skewed data like `LoanAmount` and `TotalIncome`.
- Encoded categorical variables using `LabelEncoder`.
- Standardized the feature data using `StandardScaler`.

### Machine Learning Models

Two classification algorithms were used to predict loan eligibility:

1. **Decision Tree Classifier**: 
   - Accuracy: 70.73%
   ```python
   from sklearn.tree import DecisionTreeClassifier 
   DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
   DTClassifier.fit(x_train, y_train)
   y_pred = DTClassifier.predict(x_test)
   print(f'The accuracy of decision tree is: {metrics.accuracy_score(y_test, y_pred)}')
   ```
2. Naive Bayes Classifier:
   - Accuracy: Based on testing data.
     ````python
     from sklearn.naive_bayes import GaussianNB
     NBClassifier = GaussianNB()
     NBClassifier.fit(x_train, y_train)
     y_pred = NBClassifier.predict(x_test)
     print(f'The accuracy of Naive Bayes is: {metrics.accuracy_score(y_test, y_pred)}')
     ````

### Predictions on Test Data
The model was tested on a separate test dataset (loan_test_data.csv), and the Naive Bayes classifier was used to predict loan eligibility for new applicants.
  ````python
  pred = NBClassifier.predict(test)
  ````

### Libraries Used
  - **pandas**: for data manipulation and analysis.
  - **numpy**: for numerical computations.
  - **matplotlib**: for data visualization.
  - **scikit-learn**: for machine learning models and metrics.

### How to Run
1. Clone the repository or download the project files.
2. Install the required Python libraries:
  ````bash
  pip install pandas numpy scikit-learn matplotlib
  ````
3. Run the Jupyter Notebook file **Loan Eligibility Prediction.ipynb**


### Results
The project provides a simple yet effective approach to predicting loan eligibility using Decision Tree and Naive Bayes models. These models can be further tuned and optimized for better performance.

### Conclusion
This project demonstrates how to preprocess data, build machine learning models, and evaluate their performance in predicting loan eligibility. The accuracy of the models can be improved with more advanced techniques like hyperparameter tuning or using more complex algorithms.

