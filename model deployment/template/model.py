import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

dataset= pd.read_csv('C:\\model deployment\\heart.csv')

dataset
dataset.describe()
dataset.dropna()
dataset.shape
corr_mat = dataset.corr()
sns.heatmap(corr_mat , annot=True)
plt.show()
dataset.isna().sum()

X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

X[0]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

x_train.shape
x_test.shape

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train[0]

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have X_train, X_test, y_train, y_test

# Define base classifiers
base_classifiers = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('nb', GaussianNB()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

# Define meta-classifier
meta_classifier = LogisticRegression()

# Create stacking classifier
model = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)

# Train the stacking classifier
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
from sklearn.metrics import precision_score, recall_score, f1_score



# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

with open('C:\\model deployment\\template\\stacking_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('C:\\model deployment\\template\\stacking_model.pkl', 'wb') as file:
    pickle.dump(model, file)










