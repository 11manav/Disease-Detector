
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
def predict1(request):
    return render(request, 'predict1.html')

def result1(request):
    data = pd.read_csv(r"static/Breast_train.csv")
    x = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    reg = LogisticRegression(max_iter=10000)
    reg.fit(np.nan_to_num(x_train), np.nan_to_num(y_train))

    mean_radius=request.GET['mean_radius']
    mean_texture=request.GET['mean_texture']
    mean_perimeter=request.GET['mean_perimeter']
    mean_area=request.GET['mean_area']
    mean_smoothness=request.GET['mean_smoothness']

    pred = reg.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])

    if pred == [1]:
        value = 'have breast cancer'
    else:
        value = "don't have breast cancer"

    return render(request, 'congratulations.html',
                  {
                      'context': value,
                      'diabetes': True
                  })
