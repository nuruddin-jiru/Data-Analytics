import pandas as pd
import numpy as np
#import re
import sklearn

from sklearn.ensemble import RandomForestClassifier


train=pd.read_csv("C:\\Users\\Master\\Desktop\\train.csv")
test=pd.read_csv("C:\\Users\\Master\\Desktop\\test.csv")
full_data=[train,test]

PassengerId=test['PassengerId']
#t=train.head(3)
#feature scaling and engineering
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked']=dataset['Embarked'].map({'Q': 0, 'C': 1, 'S': 2} ).astype(int)

#Dropping the features not required
train=train.drop(["Name", "PassengerId","Ticket","SibSp","Ticket","Cabin"],axis=1)
test=test.drop(["Name", "PassengerId","Ticket","SibSp","Ticket","Cabin"],axis=1)
full_data=[train,test]

#filling Fare with median value
train['Fare'].fillna(train['Fare'].dropna().median(), inplace=True)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test['Age'].fillna(test['Age'].dropna().mean(),inplace=True)
train['Age'].fillna(test['Age'].dropna().mean(),inplace=True)



#Prediction
X_train=train
X_train=X_train.drop(["Survived"],axis=1)
Y_train=train['Survived']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
test = scaler.transform(test)

#random forrest
random_forest = RandomForestClassifier(n_estimators=200,criterion='entropy',random_state=0)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test)
submission = pd.DataFrame({'PassengerId':PassengerId,'Survived':Y_pred})
filename="predictions5.csv"
submission.to_csv(filename,index=False)


