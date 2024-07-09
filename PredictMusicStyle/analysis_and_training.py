#!/usr/bin/env python3 

# Henrique Medeiros Dos Reis
# 07/09/2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# We will use the category valence to see if the musing is "slow" or "agitated"

def valence_cat(row):
    """
        Function used to categorize valance, in two different categories.

        Input: row
        Output: agitated or slow 
    """
    if row['valence'] > 0.5:
        return 'agitated'
    else:
        return 'slow'

# Create a new column based on the function
df = pd.read_csv('spotify_dataset.csv')
df['target'] = df.apply(valence_cat, axis=1)

# Let's get rid of the columns we won't be using
df_music = df.drop(['Unnamed: 0', 'track_id'],axis=1)

# Now, let's encode the colums that have are not numbers to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col_to_encode = ['artists', 'album_name', 'track_name', 'explicit', 'track_genre', 'target']
for col in col_to_encode:
    if col in df.columns:
        df_music[col] = le.fit_transform(df_music[col])
    else:
        print(f"\033[91m'Column not in dataframe\033[0m")
# sanity check
# print(df_music.head())

standardized_data = (df_music - np.mean(df_music, axis=0)) / np.std(df_music, axis=0)
cov_matrix = 1/standardized_data.shape[0]*(np.transpose(standardized_data) @ standardized_data)
stddevs = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(stddevs,stddevs)
print(corr_matrix)
# or, we can do the same using a function
print(df_music.corr())

# separate into train and test
from sklearn.model_selection import train_test_split
X = df_music[['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'track_genre']]
y = df_music['target']

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.2, shuffle=False)
# Further split into train and validation sets
X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=0.5, shuffle=False)
print('Dimensions of our splitted data')
print('X_train: ', X_train.shape, 'X_test: ',X_test.shape, 'X_val: ',X_val.shape)
print('y_train: ', y_train.shape, 'y_test: ',y_test.shape, 'y_val: ',y_val.shape)
print(df_music.shape)

X_val.to_csv('X_val.csv')
y_val.to_csv('y_val.csv')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train) 
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

# Now, lets get to the actual training
# Since I want to test different models, let's make a function to run models
def run_model(model):

    from sklearn.metrics import roc_curve, roc_auc_score, classification_report

    # run the model using the fit function
    model.fit(X_train, y_train)

    # predic probabilities and show AUC scores
    prob_predic = model.predict_proba(X_test)  
    auc = roc_auc_score(y_test, prob_predic[:,1])
    print(f"AUC {auc}")

    # make prediction and classification report
    my_prediction = model.predict(X_test)
    print("\nClassification Report")
    print(classification_report(y_test, my_prediction))

    print("\nRoc Curve\n")
    y_pred_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    # Calculate AUC 
    auc = roc_auc_score(y_test, y_pred_probs)

    # create pretty plot 
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})') # linewidth
    plt.plot([0, 1], [0, 1], color='gray',linestyle='--')
    plt.xlabel('false positive')
    plt.ylabel('true positive')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

    # make probabilites be labels
    y_pred = (y_pred_probs > 0.5).astype(int)

# Now, let's run a model
print('Logistic Regression:')
from sklearn.linear_model import LogisticRegression # natural first choice for 0 or 1 output
run_model(LogisticRegression())
print('KNN:')
from sklearn.neighbors import KNeighborsClassifier 
run_model(KNeighborsClassifier(n_neighbors=5))
print('Random Forest:')
from sklearn.ensemble import RandomForestClassifier
run_model(RandomForestClassifier(max_depth=7, n_estimators=100))

# Then we can tune parameters with Grid Search (probably not the best approach,
# but does the job)

#from sklearn.model_selection import GridSearchCV

#param_grid = {
#        "n_estimators": list(range(100, 301, 50)),
#        "max_depth":list(range(5,16))
#}

#grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1',n_jobs=1)
#grid_search.fit(X_train,y_train)
#print("Best Hyperparameters:", grid_search.best_params_)
# turned our to be 15 and 300, we can now comment the previous part in order to run it faster

print('Tunned Random Forest')
to_save_mod = RandomForestClassifier(max_depth=15,
    n_estimators=300)
run_model(to_save_mod)
joblib.dump(to_save_mod, 'random_forest_tunned_model.pkl')


