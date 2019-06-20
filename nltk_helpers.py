# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:53:03 2019

@author: S534595
"""

import nltk


from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def get_sentiments(text):
    
    analyzer=SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def split_sentiments(sentiments):
    xs=[sent['neg'] for sent in sentiments]
    ys=[sent['pos'] for sent in sentiments]
    zs=[sent['neu'] for sent in sentiments]
    return xs,ys,zs
    