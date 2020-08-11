# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:05:59 2019

@author: somay garg
"""

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]' , ' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

#Creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#Training
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.30,random_state=0)

"""
#Decomposition
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
train_x=pca.fit_transform(train_x)
test_x=pca.transform(test_x)
"""
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(train_x,train_y)

y_pred=classifier.predict(test_x)

from sklearn.metrics import accuracy_score,confusion_matrix
a_s=accuracy_score(y_pred,test_y)

cm=confusion_matrix(y_pred,test_y)