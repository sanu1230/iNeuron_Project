import pandas as pd
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
df=pd.read_csv("New Data created.csv")
df = df.sample(frac=1).reset_index(drop=True) #shuffel
print(df.head())
#train data
X=df.iloc[:,1:df.shape[1]-1]
y=df.Eligibility
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf_model=RandomForestClassifier()
clf_model.fit(X_train,y_train)
y_pred=clf_model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
