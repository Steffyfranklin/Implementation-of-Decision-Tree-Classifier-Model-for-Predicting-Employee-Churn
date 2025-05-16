# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Steffy Aavlin Raj.F.S
RegisterNumber: 212224040330
*/
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()

print("data.info()")
df.info()

print("data.isnull().sum()")
df.isnull().sum()

print("data value counts")
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()

print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()
```
## Output:
![decision tree classifier model](sam.png)

![image](https://github.com/user-attachments/assets/d74f3246-3751-4995-ace9-5a9e1a23e209)

![image](https://github.com/user-attachments/assets/22452f78-62ea-4a93-ab69-ba2c3d9bf037)

![image](https://github.com/user-attachments/assets/a7203e4c-2611-4ddd-8a98-659ee5a6af61)

![image](https://github.com/user-attachments/assets/66f98798-d026-463e-8dd0-b2379bda203f)

![image](https://github.com/user-attachments/assets/be80028e-67a1-4c78-bddd-a2682c9183d1)

![image](https://github.com/user-attachments/assets/5ecf24a5-ee9c-4da7-9c7a-5904d797a78f)

![image](https://github.com/user-attachments/assets/f791b703-981a-42dc-a45a-12ed3f670a9e)

![image](https://github.com/user-attachments/assets/9f710c9d-21be-4d5e-8b50-8f26f04ffd71)

![image](https://github.com/user-attachments/assets/57363267-24d5-4fbf-a441-af5358fad627)

![image](https://github.com/user-attachments/assets/34a6c2cc-4fc8-43a7-ac86-ac0c0fb209df)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
