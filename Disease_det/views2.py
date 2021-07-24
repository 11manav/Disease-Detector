
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def home(request):
    return render(request, 'home.html')
def predict2(request):
    return render(request, 'predict2.html')
def result2(request):
    data = pd.read_csv(r"static/Heart_train.csv")
    x = data.drop("target", axis=1)
    y = data["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    reg = LogisticRegression(max_iter=10000)

    reg.fit(np.nan_to_num(x_train), np.nan_to_num(y_train))

    age = float(request.GET['age'])
    sex = float(request.GET['sex'])
    cp = float(request.GET['cp'])
    trestbps = float(request.GET['trestbps'])
    chol = float(request.GET['chol'])
    fbs = float(request.GET['fbs'])
    restecg = float(request.GET['restecg'])
    thalach = float(request.GET['thalach'])
    exang = float(request.GET['exang'])
    oldpeak = float(request.GET['oldpeak'])
    slope = float(request.GET['slope'])
    ca = float(request.GET['ca'])
    thal = float(request.GET['thal'])

    pred=reg.predict([[age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal]])

    if pred==[1]:
        value = 'have heart '
    else:
        value = "don't have heart"

    return render(request,'congratulations.html',
                  {
                      'context' : value,
                      'diabetes' : True
                  })
