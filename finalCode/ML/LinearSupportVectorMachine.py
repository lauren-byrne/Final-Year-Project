#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_score


# In[2]:


df = pd.read_csv("test2.csv")


# In[3]:


df.head(100)


# In[4]:


# MINMAX NORMALISER
def normalise(col):
    # Create x, where x the 'scores' column's values as floats
    df[[col]] = df[[col]].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(df[[col]])

    # Run the normalizer on the dataframe
    df[[col]] = pd.DataFrame(x_scaled)    


# In[5]:


normalise('Blush')
normalise('20_y')
normalise('21_y')
normalise('22_y')
normalise('23_y')
normalise('24_y')
normalise('25_y')
normalise('26_y')


# In[6]:


df.head(5)


# In[7]:


# -LABEL ENCODING-
#encoder = preprocessing.LabelEncoder()
#df['Gaze'] = encoder.fit_transform(df['Gaze'])
#df['Blink'] = encoder.fit_transform(df['Blink'])
#df['Brows'] = encoder.fit_transform(df['Brows'])
#df['QuestionType'] = encoder.fit_transform(df['QuestionType'])


# In[8]:


# ONE HOT ENCODING
df = pd.concat([df,pd.get_dummies(df['Gaze'], prefix='Gaze')],axis=1).drop(['Gaze'],axis=1)
df = pd.concat([df,pd.get_dummies(df['Blink'], prefix='Blink')],axis=1).drop(['Blink'],axis=1)
df = pd.concat([df,pd.get_dummies(df['Brows'], prefix='Brows')],axis=1).drop(['Brows'],axis=1)
df = pd.concat([df,pd.get_dummies(df['QuestionType'], prefix='QuestionType')],axis=1).drop(['QuestionType'],axis=1)


# In[9]:


df.head(5)


# In[10]:


test = df[['QuestionType_lie', 'Gaze_center', 'Gaze_left', 'Gaze_right', 'Blink_blink', 'Blink_no blink', 'Brows_high', 'Brows_low', 'Brows_normal', 'Blush', '20_y', '21_y', "22_y","23_y", "24_y", "25_y", "26_y"]]


# In[11]:


# Pearson Correlation

plt.figure(figsize=(12,10))
cor = test.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[12]:


# FINDING FEATURE CORRELATION

cor_target = abs(cor["QuestionType_lie"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.025]
relevant_features


# In[13]:


#target Feature
targetFeature = df['QuestionType_lie']


# In[14]:


#descriptive features
X = df[['Gaze_center', 'Gaze_left', 'Gaze_right', 'Blink_blink', 'Blink_no blink', 'Brows_high', 'Brows_low', 'Brows_normal', 'Blush', '20_y', '21_y', "22_y","23_y", "24_y", "25_y", "26_y"]]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, targetFeature, test_size=0.3, random_state=109)


# In[16]:


#Feature Importance

model = ExtraTreesClassifier()
model.fit(X,targetFeature)
#print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(16).plot(kind='barh')
plt.show()


# In[17]:


#Create a LINEAR SVM Classifier
from sklearn.svm import LinearSVC
classifier = LinearSVC(C=0.1, dual = False) # Linear Kernel

#Train the model using the training sets
classifier.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[18]:


confusion_matrix(y_test, y_pred)


# In[19]:


print(classification_report(y_test, y_pred))


# In[20]:


# CROSS VALIDATION
scores = cross_val_score(classifier, X_train, y_train, cv=5)
scores


# In[21]:


# CROSS VALIDATION
cv = np.mean(cross_val_score(classifier, X_train, y_train, cv=10))
print ("Accuracy using RF with 10 cross validation : {}%".format(round(cv*100,2)))
y_predict_test = classifier.predict(X_test)

#F1_score

score_test = metrics.f1_score(y_test, y_predict_test, 
                              pos_label=list(set(y_test)), average = None)


score_test


# In[ ]:




