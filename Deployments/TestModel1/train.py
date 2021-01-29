
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from azureml.core import Workspace, Dataset, Run
import azureml.core
import random

def IsPositive(r):
    return True if r[-1] else False
def IsNegative(r):
    return True if not r[-1] else False

subscription_id = '9d0dfa04-d2f8-4521-b945-b3a7dbf43946'
resource_group = 'CougsInAzure'
workspace_name = 'CougsInAzure'

ws = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(ws, name='Alpha_Featurized')
df = dataset.to_pandas_dataframe()

df_list = df.values.tolist()


positive_dataset = list(filter(IsPositive, df_list))
negative_dataset = list(filter(IsNegative, df_list))

random.shuffle(positive_dataset)
random.shuffle(negative_dataset)

dataset = []

l = len(negative_dataset) if len(negative_dataset) < len(positive_dataset) else len(positive_dataset)

for i in range(0,l):
    dataset.append(positive_dataset[i])
    dataset.append(negative_dataset[i])

X = list(map(lambda x: x[:-1], dataset))
y = list(map(lambda x: x[-1], dataset))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

run = Run.get_context()

dt = DecisionTreeClassifier().fit(X_train, y_train)

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

#run.log('regularization rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/TestModel1.pkl')
