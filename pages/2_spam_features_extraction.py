import imp
from time import time
import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import ClassTextPreprocessing as tp
import time

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

from os.path import exists
file_exists = exists('Text_Preprocessing_New.csv')
st.set_page_config(page_title="Spam Preprocessing", layout="wide")
# st.set_page_config(layout="wide")
st.markdown("# Spam Preprocessing")

if not file_exists:
    st.subheader('Show Dataset : ')
    tweet_list = pd.read_csv("dataset_spam2.csv")
    st.write(tweet_list.head())

    ## Casefolding
    # st.header('Dataset after Casefolding Process  : ')
    # st.write('remove tweet special, number, punctuation, whitespace')
    # preprocessing = tp.OurPreprocessing()
    # tweet_list['casefolding'] = tweet_list['text'].pipe(preprocessing.process_casefolding)
    # tweet_list = tweet_list[['SPAM', 'text','casefolding']]
    # st.write(tweet_list['casefolding'].head())

    # ## Tokenizing
    # st.header('Dataset after Word Tokenizing  : ')
    # tweet_list['tweet_tokens'] = tweet_list['casefolding'].apply(preprocessing.word_tokenize_wrapper)
    # st.write(tweet_list['tweet_tokens'].head())

    # ## Frequency of Word Token
    # st.header('Frequency of Word Token  : ')
    # tweet_list['tweet_tokens_fdist'] = tweet_list['tweet_tokens'].apply(preprocessing.freqDist_wrapper)
    # st.write(tweet_list['tweet_tokens_fdist'].head())

    # ## Normalization
    # st.header('Normalization - Stopwords Removal : ')
    # tweet_list['tweet_stopwords_removal'] = tweet_list['tweet_tokens'].apply(preprocessing.stopwords_removal) 
    # st.write(tweet_list['tweet_stopwords_removal'].head())

    # st.header('Normalization - Remove Kata Gaul atau singkatan dalam bahasa indonesia : ')
    # st.write('contoh : yg (yang), knp (kenapa), dll')
    # tweet_list['tweet_normalized'] = tweet_list['tweet_stopwords_removal'].apply(preprocessing.normalized_term)
    # st.write(tweet_list['tweet_normalized'].head())

    # # st.write(%time)
    # start_time = time.time()
    # st.header('Stemming - Mengubah kata tidak baku menjadi baku dalam bahasa indonesia : ')
    # st.write('Menggunakan library Sastrawi')
    # # create stemmer
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()

    # # stemmed
    # def stemmed_wrapper(term):
    #     return stemmer.stem(term)

    # term_dict = {}

    # for document in tweet_list['tweet_normalized']:
    #     for term in document:
    #         if term not in term_dict:
    #             term_dict[term] = ' '
                

    # for term in term_dict:
    #     term_dict[term] = stemmed_wrapper(term)
        
    # # apply stemmed term to dataframe
    # def get_stemmed_term(document):
    #     return [term_dict[term] for term in document]

    # tweet_list['tweet_tokens_stemmed'] = tweet_list['tweet_normalized'].swifter.apply(get_stemmed_term)
    # st.write(tweet_list['tweet_tokens_stemmed'].head())
    # st.write("--- %s seconds ---" % (time.time() - start_time))

    # tweet_list.to_csv("Text_Preprocessing_New.csv")
else:
    st.subheader('Show Dataset : ')
    tweet_list = pd.read_csv("Text_Preprocessing_New.csv")
    st.write(tweet_list['text'].head())

    ## Casefolding
    st.header('Dataset after Casefolding Process  : ')
    st.write('remove tweet special, number, punctuation, whitespace')
    st.write(tweet_list['casefolding'].head())

    # ## Tokenizing
    st.header('Dataset after Word Tokenizing  : ')
    st.write(tweet_list['tweet_tokens'].head())

    ## Normalization
    st.header('Normalization - Stopwords Removal : ') 
    st.write(tweet_list['tweet_stopwords_removal'].head())

    st.header('Normalization - Remove Kata Gaul atau singkatan dalam bahasa indonesia : ')
    st.write('contoh : yg (yang), knp (kenapa), dll')
    st.write(tweet_list['tweet_normalized'].head())

    st.header('Stemming - Mengubah kata tidak baku menjadi baku dalam bahasa indonesia : ')
    st.write('Menggunakan library Sastrawi')
    st.write(tweet_list['tweet_tokens_stemmed'].head())






