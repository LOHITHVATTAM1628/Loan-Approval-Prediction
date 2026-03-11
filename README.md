# Loan Approval Prediction using Random Forest

## Project Overview

This project predicts whether a loan application will be approved or rejected using a machine learning model. The model is built using the Random Forest algorithm and trained on a dataset containing applicant information such as income, education, credit history, and loan amount.

The goal of this project is to demonstrate how machine learning can assist financial institutions in making data-driven loan approval decisions.

---

## Dataset

The dataset contains information about loan applicants. It includes various features that help determine whether a loan should be approved.

Example features in the dataset:

* Gender
* Married
* Dependents
* Education
* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term
* Credit_History
* Property_Area

Target variable:

Loan_Status

* 1 → Loan Approved
* 0 → Loan Rejected

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest Algorithm

---

## Machine Learning Workflow

1. Load the dataset
2. Clean the data and remove missing values
3. Convert categorical data (text) into numerical format
4. Split the dataset into training and testing data
5. Train the Random Forest model
6. Make predictions on test data
7. Evaluate the model using accuracy score

---

## Project Structure

Loan_Approval_Project
│
├── loan_data.csv
├── main.py
└── README.md

---

## How to Run the Project

Step 1: Install required libraries

pip install pandas numpy scikit-learn

Step 2: Run the program

python main.py

The program will train the model and display the prediction accuracy.

---

## Example Output

Model Accuracy: 0.81

---

## Key Learning Outcomes

* Understanding data preprocessing
* Handling missing values
* Encoding categorical variables
* Training machine learning models
* Evaluating model performance
* Implementing Random Forest for classification

---

## Future Improvements

* Add data visualization
* Implement feature importance analysis
* Build a web interface using Flask
* Deploy the model on cloud platforms

---

## Author

Vanga Reenamadhuri
Computer Science Student | Machine Learning Enthusiast | AWS Learner

---
