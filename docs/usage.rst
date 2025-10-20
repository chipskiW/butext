=======
Usage
=======



Here is an example of importing the proper libraries and functions:

Importing Necessary Packages
--------------------------------------
.. code-block :: python
	import butext as bax
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.svm import SVC
	from sklearn.metrics import classification_report
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


Uploading Datset
----------------
.. code-block :: python

   spam = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/HARP-210-NLP/main/datasets/SMSSpamCollection.tsv", sep="\t")
   spam['doc_id'] = range(len(spam)) #Need to tokenize per email, so add index column to data
   spam.head()

Output

.. code-block :: python

      class 	text	                                           doc_id
   0	 ham  	Go until jurong point, crazy.. Available only ...	0
   1	 ham	   Ok lar... Joking wif u oni...	1
   2	 spam 	Free entry in 2 a wkly comp to win FA Cup fina...	2
   3	 ham	   U dun say so early hor... U c already then say...	3
   4	 ham	   Nah I don't think he goes to usf, he lives aro...	4



Tokenizing Text
---------------
.. code-block :: python

   #Toeknize the data
   tokens = (
       spam
       .pipe(bax.tokenize,'text')
   )
   tokens

Output 

.. code-block :: none
	
      class	text	                                          doc_id  word
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  go
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  until
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  jurong
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  point
   0	ham	Go until jurong point, crazy.. Available only ...	 0	  crazy


TF-IDF
---------------

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

	X = spam_tfidf.pivot(index="doc_id", columns="word", values="tf_idf").fillna(0) #Convert into matrix format for sklearn
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

Output

.. code-block :: python
	
			 precision   recall    f1-score      support

	         ham       0.99      0.99      0.99       956
    	    	spam       0.95      0.92      0.94       156

	    accuracy                           0.98      1112
	   macro avg       0.97      0.96      0.96      1112
	weighted avg       0.98      0.98      0.98      1112


