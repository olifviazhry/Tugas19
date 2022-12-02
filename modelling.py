import streamlit as st

# for ML model
import numpy as np 
import pandas as pd
from math import sqrt
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 



st.title('Tugas 19')
st.write("""
Mencoba Modelling untuk data Heart Disease
""")

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    heart = pd.read_csv(uploaded_file)

    
    X = heart.loc[:, heart.columns != 'HeartDisease']
    y = heart["HeartDisease"]
    
#proses klasifikasi 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=60-40)
lr = LogisticRegression()
lr = lr.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

classifier_rf.fit(X_train, y_train)
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
from sklearn.model_selection import GridSearchCV

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train, y_train)

grid_search.best_score_
rf_best = grid_search.best_estimator_
rf_best.fit(X_train, y_train)
y_lr = lr.predict(X_test)
y_rf = rf_best.predict(X_test)

pca = PCA(2)
X_projected = pca.fit_transform(X)
                                      
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
                                    
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Predicted Heart Disease')
plt.ylabel('Actual Heart Disease')                                      
plt.colorbar()
                                      
st.pyplot(fig)   
    
