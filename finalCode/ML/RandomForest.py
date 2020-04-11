#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("test2.csv")


# In[3]:


df.head(5)


# In[4]:


df = pd.concat([df,pd.get_dummies(df['Gaze'], prefix='Gaze')],axis=1).drop(['Gaze'],axis=1)
df = pd.concat([df,pd.get_dummies(df['Blink'], prefix='Blink')],axis=1).drop(['Blink'],axis=1)
df = pd.concat([df,pd.get_dummies(df['Brows'], prefix='Brows')],axis=1).drop(['Brows'],axis=1)
df = pd.concat([df,pd.get_dummies(df['QuestionType'], prefix='QuestionType')],axis=1).drop(['QuestionType'],axis=1)


# In[5]:


# Create x, where x the 'scores' column's values as floats
df[['Blush']] = df[['Blush']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(df[['Blush']])

# Run the normalizer on the dataframe
df[['Blush']] = pd.DataFrame(x_scaled)


# In[6]:


df


# In[7]:


y = df['QuestionType_truth']


# In[8]:


X = df[['20_y', '21_y', '22_y','23_y', '24_y', '25_y', '26_y','Blush', 'Gaze_center', 'Gaze_left', 'Gaze_right', 'Blink_blink','Blink_no blink', 'Brows_high', 'Brows_low', 'Brows_normal' ]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)


# In[9]:



clf=RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[10]:


from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[11]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='macro')


# In[12]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[13]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[14]:


import numpy as np
from sklearn.model_selection import cross_val_score


cv = np.mean(cross_val_score(clf, X_train, y_train, cv=10))
print ("Accuracy using RF with 10 cross validation : {}%".format(round(cv*100,2)))
y_predict_test = clf.predict(X_test)

#F1_score

score_test = metrics.f1_score(y_test, y_predict_test, 
                              pos_label=list(set(y_test)), average = None)


score_test


# In[ ]:




