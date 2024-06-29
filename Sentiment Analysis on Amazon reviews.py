#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re


# In[25]:


data = pd.read_csv(r"D:\admin\Downloads\amazon_alexa.tsv", delimiter = '\t', quoting = 3)

print(f"Dataset shape : {data.shape}")


# In[26]:


data.head()


# In[27]:


data.isnull().sum()


# In[28]:


data['length'] = data['verified_reviews'].apply(len)
data.head()


# In[ ]:





# In[29]:


data.dtypes


# In[30]:


print(f"Rating value count: \n{data['rating'].value_counts()}")


# In[31]:


data['rating'].value_counts().plot.bar(color = 'red')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()


# In[32]:


fig = plt.figure(figsize=(7,7))

colors = ('red', 'green', 'blue','orange','yellow')

wp = {'linewidth':1, "edgecolor":'black'}

tags = data['rating'].value_counts()/data.shape[0]

explode=(0.1,0.1,0.1,0.1,0.1)

tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of rating')

from io import  BytesIO

graph = BytesIO()

fig.savefig(graph, format="png")


# In[33]:


print(f"Feedback value count: \n{data['feedback'].value_counts()}")


# In[34]:


review_0 = data[data['feedback'] == 0].iloc[1]['verified_reviews']
print(review_0)


# In[35]:


print(f"Feedback value count - percentage distribution: \n{round(data['feedback'].value_counts()/data.shape[0]*100,2)}")


# In[36]:


#Feedback = 0
data[data['feedback'] == 0]['rating'].value_counts()


# In[37]:


#Feedback = 1
data[data['feedback'] == 1]['rating'].value_counts()


# In[38]:


print(f"Variation value count: \n{data['variation'].value_counts()}")


# In[39]:


print(f"Variation value count - percentage distribution: \n{round(data['variation'].value_counts()/data.shape[0]*100,2)}")


# In[40]:


cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(data.verified_reviews)


# In[41]:


neg_reviews = " ".join([review for review in data[data['feedback'] == 0]['verified_reviews']])
neg_reviews = neg_reviews.lower().split()

pos_reviews = " ".join([review for review in data[data['feedback'] == 1]['verified_reviews']])
pos_reviews = pos_reviews.lower().split()

#Finding words from reviews which are present in that feedback category only
unique_negative = [x for x in neg_reviews if x not in pos_reviews]
unique_negative = " ".join(unique_negative)

unique_positive = [x for x in pos_reviews if x not in neg_reviews]
unique_positive = " ".join(unique_positive)


# #### Preprocessing and Modelling

# In[42]:


corpus = []
stemmer = PorterStemmer()
for i in range(0, data.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)


# In[43]:


cv = CountVectorizer(max_features = 2500)

#Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values


# In[44]:


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")


# In[46]:


print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")


# In[47]:


scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)


# In[48]:


model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)


# In[49]:


print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))


# In[50]:


y_preds = model_rf.predict(X_test_scl)


# In[51]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)


# In[52]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
cm_display.plot()
plt.show()


# In[53]:


accuracies = cross_val_score(estimator = model_rf, X = X_train_scl, y = y_train, cv = 10)

print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())


# In[54]:


params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}


# In[55]:


cv_object = StratifiedKFold(n_splits = 2)

grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_scl, y_train.ravel())


# In[56]:


print("Best Parameter Combination : {}".format(grid_search.best_params_))


# In[57]:


print("Cross validation mean accuracy on train set : {}".format(grid_search.cv_results_['mean_train_score'].mean()*100))
print("Cross validation mean accuracy on test set : {}".format(grid_search.cv_results_['mean_test_score'].mean()*100))
print("Accuracy score for test set :", accuracy_score(y_test, y_preds))


# In[58]:


model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)


# In[59]:


print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))


# In[60]:


y_preds = model_xgb.predict(X_test)


# In[61]:


cm = confusion_matrix(y_test, y_preds)
print(cm)


# In[62]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
cm_display.plot()
plt.show()


# ### Decision Tree Classifier

# In[63]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)


# In[64]:


print("Training Accuracy :", model_dt.score(X_train_scl, y_train))
print("Testing Accuracy :", model_dt.score(X_test_scl, y_test))


# In[65]:


y_preds = model_dt.predict(X_test)


# In[66]:


cm = confusion_matrix(y_test, y_preds)
print(cm)


# In[67]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
cm_display.plot()
plt.show()


# In[88]:



import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

# Example of input text data
new_texts = [
    "This product is amazing and exceeded my expectations!",
    "The quality of this product is good ."
]

# Text preprocessing similar to the training data
corpus_input = []
stemmer = PorterStemmer()
for text in new_texts:
    review = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    review = review.lower().split()  # Convert to lowercase and split into words
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]  # Stemming and remove stopwords
    review = ' '.join(review)  # Join the words back into a single string
    corpus_input.append(review)  # Add processed review to corpus

# Transform the input using the same CountVectorizer
X_input = cv.transform(corpus_input).toarray()  # Assuming cv is the CountVectorizer fitted on training data


# In[89]:


X_input_scl = scaler.transform(X_input)


# In[90]:


# Predict sentiment labels for the input data
y_preds = model_rf.predict(X_input_scl)

# Print predicted sentiment labels
for text, sentiment in zip(new_texts, y_preds):
    print(f"Input Text: {text}")
    print(f"Predicted Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
    print()


# In[ ]:




