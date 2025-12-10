#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk 
#nltk.download('punkt')
#nltk.download('wordnet')

import numpy as np
import pandas as pd
import string

from sklearn.utils import shuffle

import re
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

import csv 
import ast


# In[4]:


data = pd.read_csv('combined_csv.csv')
#data = shuffle(data)

raw_tweets = data.content
tweets_date = data.date
tweets_user = data.user

tweets_location = [] 

for data in tweets_user: 
    loc = ast.literal_eval(data)
    tweets_location.append(loc.get("location"))
    


# In[5]:


print(tweets_user[1])
print(raw_tweets[1])
print(tweets_date[1])
print(tweets_location[1])


# In[6]:


def clean_texts(data):
    
    #data = data.replace("[^a-zA-Z#]", " ")
    
    # remove any links
    data = re.sub(r'https://\S+', '', data)
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    
    # casefolding and removing punctuations and special chars
    data = data.lower()
    data = re.sub(r"([^a-zA-Z#\s])", '', data)
    
    # tokenise
    data = word_tokenize(data)

    return data


# In[7]:


clean_tweets = [' '.join(clean_texts(text)) for text in raw_tweets]
print(clean_tweets[0])


# In[8]:


all_stopwords = stopwords.words('english')

# add more words?
stopwo = ['new', 'news', 'broadcast', '.', '\'s', ':'] 
for words in stopwo: 
    all_stopwords.append(words)


# In[9]:


def remove_stopwords(data):
    
    data = word_tokenize(data)
    # remove stopwords
    data = [word for word in data if not word in stopwords.words()]

    return data
 


# In[ ]:


cleaner_tweets = [' '.join(remove_stopwords(text)) for text in clean_tweets]


# In[ ]:


cleaner_tweets[14]


# In[ ]:


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def process(text):
    sentence = [] 
    for word in text.split():
        word = lemmatizer.lemmatize(word)
        if not(word.endswith('e')):
                word = stemmer.stem(word)
        sentence.append(word)
    return sentence
    


# In[10]:


processed_tweets = [' '.join(process(text)) for text in cleaner_tweets]
print(processed_tweets[14])


# In[11]:


data = {'Processed': processed_tweets, 
        'Raw': raw_tweets,
        'Date': tweets_date,
        'Location': tweets_location} 
df = pd.DataFrame(data) 

df.to_csv('processed_tweets.csv')


# In[ ]:





# In[ ]:




