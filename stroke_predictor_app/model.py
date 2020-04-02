import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv("stroke_data.csv")

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = data[data.stroke==0]
df_minority = data[data.stroke==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


X = df_upsampled.drop(["stroke"], axis=1)
y = df_upsampled["stroke"]

# Split data to train and test set
# Fit and train the model
# Make Prediction on test data
# Print the roc_auc_score of the model on test data set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
np.random.seed(42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print("Score", roc_auc_score(y_test, prediction))

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))



