=============
 Use Cases
=============

Basic Usage
-----------

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

**Relative Frequencyt**

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





SVM Usage Example
-----------------

**Importing Necessary Packages**

.. code-block :: python
	
	import butext as bax
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.svm import SVC
	from sklearn.metrics import classification_report
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


**Uploading Datset**

.. code-block :: python

   spam = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/HARP-210-NLP/main/datasets/SMSSpamCollection.tsv", sep="\t")
   spam['doc_id'] = range(len(spam)) #Need to tokenize per email, so add index column to data
   spam.head()

**Output**

.. code-block :: none

      class 	text	                                           doc_id
   0	 ham  	Go until jurong point, crazy.. Available only ...	0
   1	 ham	   Ok lar... Joking wif u oni...	1
   2	 spam 	Free entry in 2 a wkly comp to win FA Cup fina...	2
   3	 ham	   U dun say so early hor... U c already then say...	3
   4	 ham	   Nah I don't think he goes to usf, he lives aro...	4



**Tokenizing Text**

.. code-block :: python

   #Toeknize the data
   tokens = (
       spam
       .pipe(bax.tokenize,'text')
   )
   tokens

**Output**

.. code-block :: none
	
      class	text	                                           doc_id  	 word
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  go
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  until
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  jurong
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  point
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  crazy


**TF-IDF**


.. code-block :: python

	df = tokens[['doc_id', 'word']]
	df = df.loc[ ~df["word"].isin(ENGLISH_STOP_WORDS) ]

	spam_tfidf = (
	    df
 	   .groupby('doc_id')['word']
 	   .value_counts(normalize=True)
    	.reset_index()
    	.pipe(bax.tf_idf, col='doc_id')
	)
	x = spam_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0]

	X = spam_tfidf.pivot(index="doc_id", columns="word", values="tf_idf").fillna(0) #Convert 	into matrix format for sklearn
	y = spam.set_index("doc_id")["class"] # set y to class, as its what we want to predict

	# Make sure number of entries are the same
	common_ids = X.index.intersection(y.index)
	X = X.loc[common_ids]
	y = y.loc[common_ids]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	svm_model = SVC(kernel='linear')
	svm_model.fit(X_train, y_train)

	y_pred = svm_model.predict(X_test)
	print(classification_report(y_test, y_pred))

**Output**

.. code-block :: none
	
			 precision   recall    f1-score  support

	         ham       0.99      0.99      0.99       956
    	    	spam       0.95      0.92      0.94       156

	    accuracy                           0.98      1112
	   macro avg       0.97      0.96      0.96      1112
	weighted avg       0.98      0.98      0.98      1112


