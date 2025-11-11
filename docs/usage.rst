===============
 **Use Cases**
===============


Logistic Regression
-------------------
A Logitsitc Regression is a supervised learning model that is used for classification tasks. This model predicts the probability of classes by learning weights and biases, and using the sigmoid function to find probability. It then makes a prediciton based on this probabilty. This model increases the weights of features to make predicitons more accurate, thus it makes the model very interperatable.

TF-IDF can be used when analyzing text for this model since, like most models, it cannot undertand string inputs, only numbers, and TF-IDF is assing each word a number (weight) based of their uniqueness. This in process tells the model which words are more important to certain classes (like spam or ham emails). Since words with zero or low TF-IDF are meaningless, they won't scew with the results of this model.


**Importing Necessary Packages**

.. code-block :: python

	import butext as bax
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import classification_report

.. code-block :: python

	spam = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/HARP-210-NLP/main/datasets/SMSSpamCollection.tsv", sep="\t")
	spam['doc_id'] = range(len(spam)) #Need to tokenize per email, so add index column to data
	spam.head()


.. code-block :: none

		class	text						     doc_id
	0	ham	Go until jurong point, crazy.. Available only ...	0
	1	ham	Ok lar... Joking wif u oni...	1
	2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	2
	3	ham	U dun say so early hor... U c already then say...	3
	4	ham	Nah I don't think he goes to usf, he lives aro...	4

.. code-block :: python
	
	log_tfidf = (
	     spam
	     .pipe(bax.tokenize, 'text')
         .pipe(bax.stopwords, 'word')
    	 .groupby('doc_id')['word']
    	 .value_counts(normalize=True)
    	 .reset_index()
    	 .pipe(bax.tf_idf, col='doc_id')
	)
	x = log_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0]

.. code-block :: python

	X = log_tfidf.pivot(index="doc_id", columns="word", values="tf_idf").fillna(0) #Convert into matrix format for sklearn

.. code-block :: python

	y = spam.set_index("doc_id")["class"] # set y to class, as its what we want to predict
	
.. code-block :: python

	y = y.map({"ham": 0, "spam": 1})

.. code-block :: python

	common_ids = X.index.intersection(y.index)
	X = X.loc[common_ids]
	y = y.loc[common_ids]

.. code-block :: python

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

.. code-block :: python

	logreg = LogisticRegression(max_iter=1000)
	logreg.fit(X_train, y_train)


.. code-block :: python

	y_pred = logreg.predict(X_test)
	print(classification_report(y_test, y_pred))

.. code-block :: python

precision    recall  f1-score   support

           0       0.97      1.00      0.98       963
           1       0.97      0.79      0.87       149

    accuracy                           0.97      1112
   macro avg       0.97      0.89      0.92      1112
weighted avg       0.97      0.97      0.97      1112

.. code-block :: python

	coef_df = pd.DataFrame({
    "word": X.columns,
    "coef": logreg.coef_[0]
	}).sort_values(by="coef", ascending=False)
	coef_df.head(10)  # top spam-indicative words

.. code-block :: none

	word	coef
8392	txt	3.607208
2221	claim	3.028673
6784	reply	2.754200
5412	mobile	2.417796
7690	stop	2.376611
7154	service	2.221344
6440	prize	2.151703
8951	won	2.149550
418	18	2.006830
8523	urgent	1.952333

.. code-block :: python

	coef_df.tail(10) #top indicative ham words

.. code-block :: none

	word	coef
9214	yup	-1.286121
4258	i'll	-1.312963
3804	got	-1.331940
6884	road	-1.439586
4130	home	-1.514126
5068	lt	-1.538432
3877	gt	-1.579734
5855	ok	-1.611774
8407	u	-1.616925
4260	i'm	-1.797147


.. code-block :: python

	spam_tfidf = (
    spam
    .pipe(bax.tokenize, 'text')
    .groupby('class')['word']
    .value_counts(normalize=True)
    .reset_index()
    .pipe(bax.tf_idf, col='class')
)
	x = spam_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0]
	x
	#We can see that many word that are important to the regression appear here, showing consistency within results

.. code-block :: python
.. code-block :: python
.. code-block :: python

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











