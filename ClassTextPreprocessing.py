import ast
from cgitb import text
import numpy as np
import pandas as pd 
import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

class OurPreprocessing:
    def __init__(self,max_features=1000):
        self.max_features = max_features

    def remove_tweet_special(self,text):
        text = str(text)
        text = text.lower()
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")

    def remove_number(self,text):
        return  re.sub(r"\d+", "", text)

    #remove punctuation
    def remove_punctuation(self,text):
        return text.translate(str.maketrans("","",string.punctuation))

    def remove_whitespace_LT(self,text):
        return text.strip()

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(self,text):
        return re.sub('\s+',' ',text)

    # remove single char
    def remove_singl_char(self,text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    def process_casefolding(self,text):
        text = text.apply(self.remove_tweet_special)
        text = text.apply(self.remove_number)
        text = text.apply(self.remove_punctuation)
        text = text.apply(self.remove_whitespace_LT)
        text = text.apply(self.remove_whitespace_multiple)
        text = text.apply(self.remove_singl_char)
        return text

    # NLTK word rokenize 
    def word_tokenize_wrapper(self,text):
        return word_tokenize(text)

        # NLTK calc frequency distribution
    def freqDist_wrapper(self,text):
        return FreqDist(text)

    def stopwords_removal(self,words):
        # ----------------------- get stopword from NLTK stopword -------------------------------
        # get stopword indonesia
        list_stopwords = stopwords.words('indonesian')


        # ---------------------------- manualy add stopword  ------------------------------------
        # append additional stopword
        list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                            'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                            'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                            'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                            '&amp', 'yah','kg','rp','Rp'])

        # ----------------------- add stopword from txt file ------------------------------------
        # read txt stopword using pandas
        txt_stopword = pd.read_csv("stopwords.csv", names= ["stopwords"], header = None)

        # convert stopword string to list & append additional stopword
        list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

        # ---------------------------------------------------------------------------------------

        # convert list to dictionary
        list_stopwords = set(list_stopwords)
        return [word for word in words if word not in list_stopwords]
    # return list_stopwords
    
    def normalized_term(self,document):
        normalizad_word = pd.read_excel("normalisasi.xlsx")
        normalizad_word_dict = {}
        for index, row in normalizad_word.iterrows():
            if row[0] not in normalizad_word_dict:
                normalizad_word_dict[row[0]] = row[1] 

        self.tweet_normalized = [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]
        return self.tweet_normalized


