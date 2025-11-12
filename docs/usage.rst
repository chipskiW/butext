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

.. code-block :: none

	lass	word	tf	idf	tf_idf
	7524	spam	claim	0.006390	0.693147	0.004429
	7528	spam	prize	0.005202	0.693147	0.003606
	35	ham	gt	0.004132	0.693147	0.002864
	7536	spam	won	0.004128	0.693147	0.002861
	37	ham	lt	0.004060	0.693147	0.002814
	...	...	...	...	...	...
	5798	ham	newspapers	0.000015	0.693147	0.000010
	5797	ham	newscaster	0.000015	0.693147	0.000010
	5828	ham	non	0.000015	0.693147	0.000010
	5827	ham	noisy	0.000015	0.693147	0.000010
	5810	ham	nike	0.000015	0.693147	0.000010

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


	

** **

.. code-block :: python

	def tokPCA(df, txtcolname, title, dim, preserves):
    '''
    df: The DataFrame being tokenized and projected into the PCA

    txtcolname: The column of df that contains strings of texts to be tokenized

    title: The column that contains the titles of this text

    dim: The number of dimensions demanded for the PCA

    preserves: A list of other column names that the user would like re-added to the PCA dataframe.

    '''
    reservedf = df[[title] + preserves]

	
**Output**

** **

.. code-block :: python

	tokens = (
   	df
    .pipe(bax.tokenize, txtcolname)
    ) #Gets tokenized version of dataframe using the butext tokenization
    df_tok = tokens.loc[ ~tokens["word"].isin(ENGLISH_STOP_WORDS) ] #Removes stop-words from the 	tokenized dataframe

	

**Output**

.. code-block :: none


** **

.. code-block :: python

	
    tfidf = (
    df_tok
    .groupby(title)['word'].value_counts(normalize = True)
    .reset_index()
    .pipe(bax.tf_idf, title)
    ) #Gets TF-IDF values from tokenized dataframe
    X = tfidf.pivot(index=title, columns="word",values="tf_idf").fillna(0)
    #pivots the data and fills NaNs as to make the format correct for scikit-learn's PCA function

**Output**

.. code-block :: python

	 pca = PCA(n_components=dim)
    X_r = pca.fit(X).transform(X) #"reduced" X
    pca_df = pd.DataFrame(X_r, columns=['PC1', 'PC2'], index=X.index)
    pca_df.reset_index(inplace=True)
    pca_df = pd.merge(reservedf, pca_df, on=title)
    return(pca_df)

.. code-block :: python

	ntflx = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/class-datasets/main/datasets/netflix.csv")
	ntflx["genre"] = ""
	ntflx.loc[ntflx["genres"].str.contains("drama"), "genre"] = "drama"
	ntflx.loc[ntflx["genres"].str.contains("comedy"),"genre"] = "comedy"
	ntflx.loc[ntflx["genres"].str.contains("horror"),"genre"] = "horror"
	ntflx.loc[ntflx["genres"].str.contains("romance"),"genre"] = "romance"
	ntflx.loc[ntflx["genres"].str.contains("documentation"),"genre"] = "documentary"
	ntflx = ntflx.loc[ntflx["genre"] != ""]
	ntflx


.. code-block :: none

	id	title	type	description	release_year	age_certification	runtime	genres production_countries	seasons	imdb_id	imdb_score	imdb_votes	tmdb_popularity	tmdb_score	genre
	0	ts300399	Five Came Back: The Reference Films	SHOW	This collection includes 12 World War II-era p...	1945	TV-MA	48	['documentation']	['US']	1.0	NaN	NaN	NaN	0.600	NaN	documentary
	1	tm84618	Taxi Driver	MOVIE	A mentally unstable Vietnam War veteran works ...	1976	R	113	['crime', 'drama']	['US']	NaN	tt0075314	8.3	795222.0	27.612	8.2	drama
	2	tm127384	Monty Python and the Holy Grail	MOVIE	King Arthur, accompanied by his squire, recrui...	1975	PG	91	['comedy', 'fantasy']	['GB']	NaN	tt0071853	8.2	530877.0	18.216	7.8	comedy
	3	tm70993	Life of Brian	MOVIE	Brian Cohen is an average young Jewish man, bu...	1979	R	94	['comedy']	['GB']	NaN	tt0079470	8.0	392419.0	17.505	7.8	comedy
	4	tm190788	The Exorcist	MOVIE	12-year-old Regan MacNeil begins to adapt an e...	1973	R	133	['horror']	['US']	NaN	tt0070047	8.1	391942.0	95.337	7.7	horror
	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
	5782	tm1040816	Momshies! Your Soul is Mine	MOVIE	Three women with totally different lives accid...	2021	NaN	108	['comedy']	['PH']	NaN	tt14412240	5.8	26.0	4.112	NaN	comedy
	5783	tm1014599	Fine Wine	MOVIE	A beautiful love story that can happen between...	2021	NaN	100	['romance', 'drama']	['NG']	NaN	tt13857480	6.9	39.0	0.966	NaN	romance
	5784	tm1108171	Edis Starlight	MOVIE	Rising star Edis's career journey with ups and...	2021	NaN	74	['music', 'documentation']	[]	NaN	NaN	NaN	NaN	1.036	8.5	documentary
	5785	tm1045018	Clash	MOVIE	A man from Nigeria returns to his family in Ca...	2021	NaN	88	['family', 'drama']	['NG', 'CA']	NaN	tt14620732	6.5	32.0	0.709	NaN	drama
	5787	ts271048	Mighty Little Bheem: Kite Festival	SHOW	With winter behind them, Bheem and his townspe...	2021	NaN	0	['family', 'comedy', 'animation']	[]

.. code-block :: python

	newdf = tokPCA(ntflx,"description","id",2,["genre","age_certification"])
	newdf


.. code-block :: none

	id	genre	age_certification	PC1	PC2
		0	ts300399	documentary	TV-MA	0.029556	0.000573
		1	tm84618	drama	R	-0.004263	0.004507
		2	tm127384	comedy	PG	-0.003020	0.000247
		3	tm70993	comedy	R	-0.014535	-0.001630
		4	tm190788	horror	R	-0.039231	0.000530
		...	...	...	...	...	...
		5074	tm1040816	comedy	NaN	-0.009749	0.001129
		5075	tm1014599	romance	NaN	-0.073217	-0.018134
		5076	tm1108171	documentary	NaN	0.032761	-0.021259
		5077	tm1045018	drama	NaN	-0.049431	-0.000664
		5078	ts271048	comedy	NaN	0.023012	0.000648


Problem: Visualizing this PCA makes it clear that an outlier is skewing the data. Because PCA finds the dimensions meant to explain a lot of variance in the data, outliers can skew the dimensions and make them less useful for visualization.

.. code-block :: python

	plt.scatter(x=newdf.PC1,y=newdf.PC2,alpha=0.5)


.. image:: _build/html/_static/PCA1.png
	:alt: description
	:width: 400px

.. code-block :: python

	outlier = newdf[newdf.PC2>1]
	outlier #the value we want to get rid of
.. code-block :: none

		id		genre	    age_certification	PC1		PC2
	2299	tm375302	documentary	NaN		0.110894	4.468589

Now that we know what value is causing the discrepancy, we can re-run the function.

.. code-block :: python

	ntflx2 = ntflx[ntflx.id != "tm375302"]
	newdf2 = tokPCA(ntflx2,"description","id",2,["genre","age_certification"])
	#https://www.statology.org/matplotlib-scatterplot-color-by-value/
	groups = newdf2.groupby("genre")
	for name, group in groups:
	plt.plot(group.PC1, group.PC2, marker='o', linestyle='', markersize=4,alpha=0.7,label=name)
	plt.xlabel("PC1 (0.19%)")
	plt.ylabel("PC2 (0.16%)")
	plt.legend()
	plt.show()



.. image:: _build/html/_static/PCA2.png
	:alt: description
	:width: 400px
Judging by the PCA, certain patterns begin to emerge which can tell us about what our principal components may represent. PC1 may have to do with the "seriousness" of words, while PC2 may have to do with whether words are more emotional or descriptive.

.. code-block :: none
.. code-block :: none
.. code-block :: none
.. code-block :: none






