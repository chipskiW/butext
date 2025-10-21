===============
 **Use Cases**
===============

SVM  Example
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


PCA Example
-----------------
**Importing Necessary Packages**

.. code-block :: python

	import butext as bax
	from sklearn.decomposition import PCA
	import pandas as pd
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	import matplotlib.pyplot as plt

**Uploading Dataset**

.. code-block :: python

	ntflx = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/class-datasets/main/datasets/netflix.csv")
	ntflx

** **

.. code-block :: python

	tokens = (
    ntflx
    .pipe(bax.tokenize, 'description')
	)
	df = tokens.loc[ ~tokens["word"].isin(ENGLISH_STOP_WORDS) ]

	tfidf = (
 	   df
  	  .groupby('id')['word'].value_counts(normalize = True)
   	 .reset_index()
   	 .pipe(bax.tf_idf, 'id')
	)
	X = tfidf.pivot(index="id", columns="word",values="tf_idf").fillna(0)
	X

**Output**

** **

.. code-block :: python

	pca = PCA(n_components=2) #2-dimensional PCA
	X_red = pca.fit(X).transform(X)
	pca.explained_variance_ratio_ #percenage of variance in the data explained by PC1 and PC2 respectively

**Output**

.. code-block :: none

	array([0.00191735, 0.00177282])


** **
.. code-block :: python
	pca_df = pd.DataFrame(X_red, columns=['PC1', 'PC2'], index=X.index)
	plt.scatter(x=pca_df.PC1,y=pca_df.PC2,alpha=0.5)

**Output**

.. image:: /_static/Unknown3.png
   :alt: Message class distribution
   :align: center
   :width: 400px











