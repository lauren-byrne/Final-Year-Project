#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


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


df.head(5)


# In[6]:


targetFeature = df['QuestionType_lie']


# In[7]:


df


# In[8]:


X = df[['Gaze_center', 'Gaze_left', 'Gaze_right', 'Blink_blink', 'Blink_no blink', 'Brows_high', 'Brows_low', 'Brows_normal', 'Blush', '20_y', '21_y', "22_y","23_y", "24_y", "25_y", "26_y"]]

X_train, X_test, y_train, y_test = train_test_split(X, targetFeature, test_size=0.3, random_state=109)


# In[9]:


classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)


# In[10]:


#Predict the response for test dataset
y_pred = classifier.predict(X_test)


# In[11]:



# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[12]:




print(classification_report(y_test, y_pred))


# In[13]:


from sklearn.metrics import confusion_matrix


confusion_matrix(y_test, y_pred)



# In[14]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
scores


# In[15]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:




