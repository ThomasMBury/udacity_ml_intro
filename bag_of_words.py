#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:11:30 2018

@author: Thomas Bury
"""


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import nltk
nltk.download('stopwords')

vectorizer = CountVectorizer()

string1 = "Tom <3 Steph"
string2 = "It is really cold today"
string3 = "How long long is a piece of string?"

string_list = [string1, string2, string3]

bag_of_words = vectorizer.fit(string_list)
bag_of_words = vectorizer.transform(string_list)


# Count stopwords
sw = stopwords.words("english")

# Stemming
stemmer = SnowballStemmer('english')

