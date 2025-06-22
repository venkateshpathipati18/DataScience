#Importing dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')

# Loading Dataset

data = pd.read_csv(r"C:\Users\Thota Goutham\Desktop\Diabetes Project Files\diabetes.csv")
print("Successfully Imported Data!")
print(data.head())

print(data.shape)

# Description

print(data.describe(include='all'))

# Finding Null Values

print(data.isna().sum())

print(data.corr())

print(data.groupby('Age').mean())

print(data['Outcome'].value_counts())

#0 means no diabeted
#1 means patient with diabtes

# Data Analysis:

## Countplot:

sns.countplot(data['Age'])
plt.show()

sns.countplot(data['Pregnancies'])
plt.show()

sns.countplot(data['BMI'])
plt.show()

sns.countplot(data['Outcome'])
plt.show()

# Distplot:

sns.distplot(data['Outcome'])
plt.show()
sns.distplot(data['BMI'])
plt.show()
sns.distplot(data['Pregnancies'])
plt.show()
sns.distplot(data['Age'])
plt.show()
sns.distplot(data['BP'])
plt.show()
sns.distplot(data['ST'])
plt.show()
data.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)

data.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)

# Histogram

data.hist(figsize=(10,10),bins=50)
plt.show()

# Heatmap for expressing correlation

corr = data.corr()
sns.heatmap(corr,annot=True)
plt.show()
# Box plot for outlier visualization

sns.set(style="whitegrid")
data.boxplot(figsize=(15,6))
plt.show()
# Pairplot:

sns.pairplot(data)
plt.show()
# Violinplot:

sns.violinplot(x='Outcome', y='Age', data=data)
plt.show()
# sns.violinplot(x='Outcome', y='BloodPressure', data=data)

sns.violinplot(x='Outcome', y='ST', data=data)
plt.show()
# Pairplot:

sns.pairplot(data,hue='Outcome');
plt.show()
# Feature Selection

#lets extract features and targets
X = data.drop(columns=['Outcome'])
Y = data['Outcome']
print("Features Extraction Sucessfull")

# Feature Importance

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)

# Splitting Dataset

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# Using Logistic Regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("--Logistic Regression--")
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))

confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)

# Using KNN

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("--KNN--")
print("Accuracy Score:",accuracy_score(Y_test,y_pred))

# Using SVC

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("--SVC--")
print("Accuracy Score:",accuracy_score(Y_test,pred_y))

# Using Decision Tree

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred1 = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("--Decision Tree--")
print("Accuracy Score:",accuracy_score(Y_test,y_pred1))

# Using GaussianNB

from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("--GaussianNB--")
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))

# Random Forest

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("--Random Forest--")
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))

# Using Xgboost

import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1,eval_metric='mlogloss')
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("--Xgboost--")
print("Accuracy Score:",accuracy_score(Y_test,y_pred5))

# Results

results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [accuracy_score(Y_test,Y_pred),accuracy_score(Y_test,y_pred),accuracy_score(Y_test,pred_y),accuracy_score(Y_test,y_pred1),accuracy_score(Y_test,y_pred3),accuracy_score(Y_test,y_pred2),accuracy_score(Y_test,y_pred5)]})



result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df)





