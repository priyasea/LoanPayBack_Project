import requests

url = 'http://localhost:9696/predict'

customer = {
         "loan_amount": 15000,
         "interest_rate": 12.5,
         "credit_score": 689,
         "annual_income": 82000,
         "debt_to_income_ratio": 17.3,
        "employment_status": "student",
        "education_level": "bachelors",
        "grade_subgrade": "b3"
}

response = requests.post(url, json=customer)
predictions = response.json()

print(predictions)
if predictions.get('loan_paid_back'):
    print('customer is likely to pay back loan, Approve the Loan')
else:
    print('customer is not likely to pay back loan')