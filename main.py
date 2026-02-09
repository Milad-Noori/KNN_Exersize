import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings
data = pd.read_csv('diabetes.csv')
data.describe()
df=pd.DataFrame(data)
df
data = pd.read_csv('diabetes.csv')
data.describe()
X = pd.DataFrame(data ,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
Y = data['Outcome'].values.reshape(-1 , 1)
X_train ,X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.3 , random_state=0)
k = 5
model=KNeighborsClassifier(k)
model.fit(X_train ,Y_train.ravel())
y_pred=model.predict(X_test)
print('accuracy:',metrics.accuracy_score(Y_test ,y_pred))
k = 4
model=KNeighborsClassifier(k)
model.fit(X_train ,Y_train.ravel())
y_pred=model.predict(X_test)
print('accuracy:',metrics.accuracy_score(Y_test ,y_pred))
k = 10
model=KNeighborsClassifier(k)
model.fit(X_train ,Y_train.ravel())
y_pred=model.predict(X_test)
print('accuracy:',metrics.accuracy_score(Y_test ,y_pred))
k = 20
acc = np.zeros((k))
for i in range (1 , k+1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train , Y_train.ravel() )
    y_pred = knn.predict(X_test)
    acc[i-1]=metrics.accuracy_score (Y_test , y_pred)
acc
print(np.max(acc))
print(np.min(acc))
from sklearn.model_selection import GridSearchCV
parametrs = {'n_neighbors':range (1,50)}
grid_kn =GridSearchCV(estimator=knn,
                     param_grid=parametrs,
                     scoring='accuracy',
                     cv=5,
                     verbose=1,
                     n_jobs=-1)
grid_kn.fit(X_train ,Y_train.ravel())
grid_kn.best_params_
data.corr()
new_data = pd.DataFrame([[6,148,72,35.0,33.6,0.627,50,50]],columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
P1 =knn.predict(new_data)
P1
metrics.accuracy_score (Y_test , y_pred)
