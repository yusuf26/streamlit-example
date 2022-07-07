import ast
from cgitb import text
import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix
class OurTFIDF:
    def __init__(self,max_features=1000):
        self.max_features = max_features

    def convert_text_list(self,texts):
        texts = ast.literal_eval(texts)
        return [text for text in texts]

    def calc_TF(self,document):
        # Counts the number of times the word appears in review
        TF_dict = {}
        for term in document:
            if term in TF_dict:
                TF_dict[term] += 1
            else:
                TF_dict[term] = 1
        # Computes tf for each word
        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(document)
        return TF_dict

    def calc_DF(self,tfDict):
        count_DF = {}
        # Run through each document's tf dictionary and increment countDict's (term, doc) pair
        for document in tfDict:
            for term in document:
                if term in count_DF:
                    count_DF[term] += 1
                else:
                    count_DF[term] = 1
        return count_DF

    # n_document = len(TWEET_DATA)

    def calc_IDF(self,__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict

    def calc_TF_IDF(self,TF):
        TF_IDF_Dict = {}
        #For each word in the review, we multiply its tf and its idf.
        for key in TF:
            TF_IDF_Dict[key] = TF[key] * self.IDF[key]
        return TF_IDF_Dict

    
    def calc_TF_IDF_Vec(self,__TF_IDF_Dict):
        TF_IDF_vector = [0.0] * len(self.unique_term)

        # For each unique word, if it is in the review, store its TF-IDF value.
        for i, term in enumerate(self.unique_term):
            if term in __TF_IDF_Dict:
                TF_IDF_vector[i] = __TF_IDF_Dict[term]
        return TF_IDF_vector

    
    def join_text_list(self,texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])

    def ranking_features(self,data):
      # Convert Series to List
      TF_IDF_Vec_List = np.array(data.to_list())
      # Sum element vector in axis=0 
      sums = TF_IDF_Vec_List.sum(axis=0)
      data = []
      for col, term in enumerate(self.unique_term):
          data.append((term, sums[col]))
      ranking = pd.DataFrame(data, columns=['term', 'rank'])
      ranking.sort_values('rank', ascending=False)
      return ranking

    def process(self,tweet_list):
      tweet_list['tweet_list'] = tweet_list['tweet'].apply(self.convert_text_list)
      tweet_list["TF_dict"] = tweet_list['tweet_list'].apply(self.calc_TF)
      DF = self.calc_DF(tweet_list["TF_dict"])
      n_document = len(tweet_list)
      self.IDF = self.calc_IDF(n_document, DF)

      tweet_list["TF-IDF_dict"] = tweet_list["TF_dict"].apply(self.calc_TF_IDF)

      # sort descending by value for DF dictionary 
      sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:self.max_features]

      # Create a list of unique words from sorted dictionay `sorted_DF`
      self.unique_term = [item[0] for item in sorted_DF]
      tweet_list["TF_IDF_Vec"] = tweet_list["TF-IDF_dict"].apply(self.calc_TF_IDF_Vec)

      tweet_list["tweet_join"] = tweet_list["tweet"].apply(self.join_text_list)
      

      return tweet_list

    def fit_transform(self,tweet_list):
      output = self.process(tweet_list)
      arr_features = []
      for x in output["TF_IDF_Vec"]:
        arr_features.append(x)
      features = csr_matrix(arr_features,dtype=np.float64)
      return features

    def transform(self,tweet_list):
      output = self.fit_transform(tweet_list)
      return output.toarray()
