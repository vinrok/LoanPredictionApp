# LoanPrediction

## Binary classification (Y/N) -- On the basis of input taken from users, predicting whether user will be granted loan or not.

Accepting from the users following inputs:
- Gender
- Married
- Dependents
- Education
- Self Employed
- Loan Amount
- Applicant Income
- Coapplicant Income
- Loan Amount Term
- Credit History

When user clicks on predict, the model will perform execution and return result in the form of binary classification i.e. Y/N

For this, I used Pycaret library to perform EDA, feature importance and then model building further compared various models such as-
- GradientBoost classifier
- Logistic Regression
- RandomForest Classifier
- XGBoost, etc..

Finally, _f1-score, accuracy_ for **GradientBoost classifier** was comparatively better.
