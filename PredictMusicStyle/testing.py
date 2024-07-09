#!/usr/bin/env python3

# Henrique Medeiros Dos Reis
# 07/09/2024

from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np

loaded_model = joblib.load('random_forest_tunned_model.pkl')
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv')

X_val = X_val.drop(['Unnamed: 0'],axis=1)
print(X_val.shape)
y_val = y_val.drop(['Unnamed: 0'],axis=1)

# Since we splitted the data three ways at the begging of the experiment
# we dont need to do the exact same steps as we did with training, but here
# is what you should do in case they are not properly processed already

#column = ['track_genre']
#label_encoder_dataframe(X_val, column)
#X_val = scaler.transform(X_val)

# Now, lets apply the model
predictions = loaded_model.predict(X_val)
def get_val(predictions):
    results = []
    for i in predictions:
        if i==1:
            results.append('Slow')
        elif i==0: 
            results.append('Agitaded')
        else:
            results.append('Undefined')
    return np.array(results) 

X_val['target'] = get_val(predictions)
y_val['target'] = get_val(predictions)
print(X_val)
print(y_val)
# the following is going to count how many times we made an error 
count = 0
for i in range(len(y_val)):  # Iterate over indices instead of values directly
    if y_val.iloc[i]['target'] != X_val.iloc[i]['target']:
        count += 1
print(count)
# The results were great, 0 errors in unseen data
