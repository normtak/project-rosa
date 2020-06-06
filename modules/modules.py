# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:36:12 2020

@author: Stan
"""
from nltk.corpus import stopwords
import numpy as np

def word_extraction(sentence):
    ignore = stopwords.words()
    words = sentence.split()
    cleaned_text = [w.lower() for w in words if w not in ignore]    
    return cleaned_text

def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words
    
def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab));
    print(len(vocab))
    bag_vectors = []
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab): 
                if word == w:
                    bag_vector[i] += 1
        bag_vectors.append(bag_vector)
    return bag_vectors