================
**INTRODUCTION**
================


_a link: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#hyperlinks
This is ReStructured text syntax

Here we will explain what the project is, our goals, and how it works. 

Overview
--------

Butext focuses on text proccessing techniques often used in natural lanuage proccessing. 
We will cover:

* Tokenization 
* Relative Frequency 
* Term Frequency Inverse Document Frequency (TF-IDF)


Tokenization
------------
Tokenization is the process of breaking down text into individual words called tokens.[e.g. “I like dogs” -> [“I”, “like”, “dogs’] 
This allows the 

Tokenization Example
====================

**Importing Necessary Packages**

.. code-block :: python

	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	import butext as bax



**Uploading Datset**

.. code-block :: python
	
	netflix = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/class-				datasets/main/datasets/netflix.csv")
	netflix.head(2)

**Output**

**Tokenizing Text**

.. code-block :: python

	tokens = (
 	   netflix
    	.pipe(bax.tokenize, 'description')
	)

**Output**

.. code-block :: none
	
	id 	  title  			       type   release_year  age_certification  runtime  genres  		production_countries seasons imdb_id    imdb_score  imdb_votes tmdb_popularity  tmdb_score
	ts300399  Five Came Back: The Reference Films  SHOW   1945          TV-MA              48       ['documentation']       ['US']               1.0      NaN       NaN        NaN         0.600             NaN
	tm84618    Taxi Driver                         MOVIE  1976          R                  113      ['crime', 'drama']      ['US']               NaN      tt0075314 8.3        795222.0    27.612            8.2

Relative Frequency 
------------------

*Relative Frequency is a simple mathematical operation that divides text frequency of a single word from one document by the text frequency of the same word from a different document*

Relative Frequency Example
====================

**Relative Frequency**

.. code-block :: python

	#Want to find relative frequncy of words assocaited with tv show or movies
	df = tokens[['word', 'type']]
	df = df.loc[ ~df["word"].isin(ENGLISH_STOP_WORDS) ]

	rel_freq = (
    	df
    	.groupby('type')['word'].value_counts(normalize = True)
    	.reset_index()
    	.query('proportion > 0.0005')
    	.pipe(bax.rel_freq, 'type')
	)

**Output**

.. code-block :: none 

	type   word	  MOVIE	        SHOW	        rel_freq	 logratio
	245	series	0.000883	0.007439	0.118762	-0.925322
	71	drama	0.000250	0.001998	0.125120	-0.902672
	3   adventures	0.000250	0.001733	0.144236	-0.840926
	225	reality	0.000250	0.001468	0.170246	-0.768923
	297	tv	0.000250	0.001300	0.192315	-0.715987

Our function is dividing the text frequency of a word in movies description divided by that same word in show descriptions. So by taking a logration of the relative frequency, we can see which word is more greatly associated with with category. Since we are dividing by the text frequency of show, and since  𝑙𝑜𝑔(𝐴/𝐵)=𝑙𝑜𝑔(𝐴)−𝑙𝑜𝑔(𝐵) , then a greater negative value means more greatly associated with show, and vice versa.

.. code-block :: python

	mostfreq = pd.concat([  rel_freq[0:10] , rel_freq[-10:]  ])
	sns.barplot(data=mostfreq, x="logratio", y="word")
	plt.xlabel("Logratio")
	plt.show()

.. image:: _static/Unknown.png

   :alt: Message class distribution
   :align: center
   :width: 400px





Term-Frequency Inverse Document Frequency (TF-IDF)
-------------------------------------------------

TF-IDF allows us to measure the uniqueness of a word to a given document.

TF-IDF Example
====================

.. code-block :: python

	df = tokens2[['genre', 'word']]
	df = df.loc[ ~df["word"].isin(ENGLISH_STOP_WORDS) ]

	tfidf = (
    df
    .groupby('genre')['word'].value_counts(normalize = True)
    .reset_index()
    .pipe(bax.tf_idf, 'genre')
	)

	x = tfidf.loc[tfidf.tf_idf != 0]
	x= x.sort_values(by = 'tf_idf', ascending= False)
	x

**Output**

.. code-block :: none

		genre		word	        tf	   	idf	  	tf_idf
	10623	documentary	docuseries	0.002510	1.609438	0.004039
	10611	documentary	documentary	0.008293	0.223144	0.001851
	9         comedy        stand-up       0.003001         0.510826        0.001533
	27777	  horror	vampires	0.001597	0.916291	0.001463
	10649	documentary	interviews	0.001473	0.916291	0.001350


.. code-block :: python

	viz = x[0:10]
	sns.barplot(data = viz, x= 'word', y = 'tf_idf', hue = 'genre')
	plt.xticks(rotation = 45)
	plt.show()





