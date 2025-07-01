import numpy as np
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

data = pd.read_csv('/new_file.csv')

data.head()

data.info()

data.describe()

data.isnull().sum() #checking for null values/missing values

data['isFraud'].value_counts() #checking for class imbalance
#check 0-> not fraud, 1-> fraud

#seperating categorical and numerical variables
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

#seperating data for analysis

legit= data[data['isFraud'] == 0]
fraud = data[data['isFraud'] == 1]

print("Legit transactions:", len(legit))
print("Fraud transactions:", len(fraud))

new_dataset= pd.concat([legit.sample(), fraud], axis=0) #concatenating row wise
#axis - >0 row   axis ->1 column

new_dataset.head()

new_dataset['isFraud'].value_counts()


new_dataset.head()
new_dataset.tail()
new_dataset['isFraud'].value_counts()

#splitting data into training sets and data

X = new_dataset.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = new_dataset['isFraud']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

print("X_test","\n",X_test)

print("y_test","\n",y_test)

print("X_train","\n",X_train)

print("y_train","\n",y_train)


#training the model

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as ras
model = [XGBClassifier()]

for i in range(len(model)):
	model[i].fit(X_train, y_train)
	print(f'{model[i]} : ')

	train_preds = model[i].predict_proba(X_train)[:, 1]
	print('Training Accuracy : ', ras(y_train, train_preds))

	y_preds = model[i].predict_proba(X_test)[:, 1]
	print('Validation Accuracy : ', ras(y_test, y_preds))
	print()

 #ACCURACY



y_train_pred = model[0].predict(X_train)
y_test_pred = model[0].predict(X_test)


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
