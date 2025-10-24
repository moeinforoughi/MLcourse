#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://127.0.0.1:9696/predict'

customer = {
        "customerid": "1987-xyztq",
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "no",
        "dependents": "no",
        "tenure": 2,
        "phoneservice": "yes",
        "multiplelines": "yes",
        "internetservice": "fiber_optic",
        "onlinesecurity": "no",
        "deviceprotection": "yes",
        "techsupport": "no",
        "streamingtv": "yes",
        "streamingmovies": "yes",
        "contract": "month_to_month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "monthlycharges": 95.8,
        "totalcharges": 180.45,
        "churn": 1
}


response = requests.post(url, json=customer).json()


if response['churn'] == True:
    print(f'Sending a promotional Email to 5442-pptjy ')



