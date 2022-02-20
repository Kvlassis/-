import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("Dataset 1.csv")


features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

X = df[features]
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)



print("Score of Tree:")
print(dtree.score(X_test,y_test))
a=len(features)
nn =  MLPClassifier(random_state=0, max_iter=500,hidden_layer_sizes=(a,)).fit(X_train,y_train)

dtree = nn.fit(X_train, y_train)

print("Score of Neural Networks1:")
print(nn.score(X_test,y_test))

nn =  MLPClassifier(random_state=0, max_iter=500,hidden_layer_sizes=(2*a,a)).fit(X_train,y_train)

dtree = nn.fit(X_train, y_train)


print("Score of Neural Networks1:")
print(nn.score(X_test,y_test))

nn =  MLPClassifier(random_state=0, max_iter=500,hidden_layer_sizes=(2*a,a,int(a/2))).fit(X_train,y_train)

dtree = nn.fit(X_train, y_train)

print("Score of Neural Networks2:")
print(nn.score(X_test,y_test))


nn =  MLPClassifier(random_state=0, max_iter=500,hidden_layer_sizes=(3*a,2*a,a)).fit(X_train,y_train)

dtree = nn.fit(X_train, y_train)

print("Score of Neural Networks3:")
print(nn.score(X_test,y_test))

nn =  MLPClassifier(random_state=0).fit(X_train,y_train)

dtree = nn.fit(X_train, y_train)

print("Score of Neural Networks3:")
print(nn.score(X_test,y_test))

nn =  MLPClassifier(random_state=0, max_iter=500,hidden_layer_sizes=(2*a,a,)).fit(X_train,y_train)

dtree = nn.fit(X_train, y_train)

from sklearn import metrics
xout=nn.predict(X_test)
print(metrics.classification_report(xout,y_test))


