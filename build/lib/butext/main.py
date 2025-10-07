import re
import pandas as pd
import numpy as np
from wordcloud import STOPWORDS

def tokenize(df, col):
  #assign method adds new column; .findall creates list of words using regex, .explode creates individual rows for each word in the list
  tokens= df.assign(word= df[col].str.lower().str.findall(r"\w+(?:\S?\w+)*")).explode("word")
  return tokens   #remove ["word"] to get entire dataframe

def rel_freq(df , col, col2):
    tok = tokenize(df, col)
    df = tok[[col2, "word"]]
    df = df.loc[ ~df["word"].isin(STOPWORDS) ]
    group = df.groupby(col2)["word"].value_counts(normalize = True)
    group = group[group > 0.0005]
    group.name = "text_freq"
    group = group.reset_index()
    group = group.pivot(index='word', columns= col2, values='text_freq')
    group = group.reset_index()
    group.loc[group[group.columns[1]].isna(), group.columns[1]] = 0.0005/2
    group.loc[group[group.columns[2]].isna(), group.columns[2]]  = 0.0005/2
    group['rel_freq'] = group[group.columns[1]]/group[group.columns[2]]
    group["logratio"] = np.log10(group["rel_freq"])
    return group
# Let col be the column you want to tokenize and col2 be the categories you want to find rel freq between

def tfidf(df , col, col2):
  tok = tokenize(df, col)
  df = tok[[col2, 'word']]
  df = df.loc[ ~df["word"].isin(STOPWORDS) ]
  counts = df.groupby(col2)["word"].value_counts()
  counts.name = "n"
  counts = counts.reset_index()
  tf = df.groupby(col2)["word"].value_counts(normalize = True)
  tf.name = "text_freq"
  tf = tf.reset_index()
  counts['tf'] = tf['text_freq'].values
  doc = counts.groupby('word')[col2].count()
  doc.name = 'df'
  doc = doc.reset_index()
  N = df[col2].nunique()
  doc['idf'] = np.log(N / doc["df"])
  result = counts.merge(doc[["word", "idf"]], on="word")
  result["tfidf"] = result["tf"] * result["idf"]
  return result