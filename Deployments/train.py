#%%writefile $script_folder/train.py

import argparse
import os
import numpy as np
import glob

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

from azureml.core import Run, Workspace, Dataset

subscription_id = '9d0dfa04-d2f8-4521-b945-b3a7dbf43946'
resource_group = 'CougsInAzure'
workspace_name = 'CougsInAzure'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Alpha_Dataset_Featurized')
df = dataset.to_pandas_dataframe()

X = df["Data"].tolist()
y = df["Label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

# get hold of the current run
run = Run.get_context()

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/TestModel1.pkl')