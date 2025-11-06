================
**OVERVIEW**
================


_a link: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#hyperlinks
This is ReStructured text syntax
 

*Tokenization*
--------------
Tokenization is the process in which a particular portion of text is split into individual words or tokens. For example, the sentence "I like dogs" would become ["I" , "like", "dogs"]. While at its surface it seems like a pretty simple task, it assists with basic textual anaysis such as word counting and text frequencies all the way to Machine Learning, Natural Language Processing, and Deep Learning. In the 2017 paper "Attention Is All You Need" by Ashish Vaswani et al. which is the creation of the transfomer architecture which gave birth to the GPT decoder which is resposible for Chat-GPT, the architecture starts with tokenizing text and assign each word a vector embedding. Thus, while this task seems rudimentary, it has birthed many of the AI/Machine Learning developments in todays society.

Now we'll start with the basics of tokenization and see how it works:

**Example**
^^^^^^^^^^^

**Importing Necessary Packages**

.. code-block :: python

	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	import butext as bax



**Uploading Datset**

.. code-block :: python
	
	spam = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/HARP-210-NLP/main/datasets/SMSSpamCollection.tsv",sep="\t")
	spam.head(5)

**Output**

.. code-block :: none

		class	text
	0	ham	Go until jurong point, crazy.. Available only ...
	1	ham	Ok lar... Joking wif u oni...
	2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
	3	ham	U dun say so early hor... U c already then say...
	4	ham	Nah I don't think he goes to usf, he lives aro...


**Tokenization In Action**

Now that our data is loaded, we can use our tokenization function to find some basic information about our data. Our tokenization function is denoted by bax.tokenize and works best within pandas pipe() operator. We will also utilize our stopewords function denoted by bax.stopwords as it removes words not important to our analysis

.. code-block :: python

	spam_tokens = (
   	 	spam
    		.pipe(bax.tokenize, 'text')
    		.pipe(bax.stopwords, 'word')
		)
		spam_tokens

**Output**

.. code-block :: none
	
		class	text							word
	0	ham	Go until jurong point, crazy.. Available only ...	jurong
	0	ham	Go until jurong point, crazy.. Available only ...	point
	0	ham	Go until jurong point, crazy.. Available only ...	crazy
	0	ham	Go until jurong point, crazy.. Available only ...	available
	0	ham	Go until jurong point, crazy.. Available only ...	bugis
	...	...	... 							...
	5570	ham	The guy did some bitching but I acted like i'd...	week
	5570	ham	The guy did some bitching but I acted like i'd...	gave
	5570	ham	The guy did some bitching but I acted like i'd...	free
	5571	ham	Rofl. Its true to its name				rofl
	5571	ham	Rofl. Its true to its name				true

Now that we have out tokens, we can find plenty of useful information. Lets start with the most common words.

.. code-block :: python
	
	spam_tokens['word'].value_counts()

.. code-block :: none

		
	word    count	
	u	1140
	2	482
	i'm	393
	ur	390
	just	371
	...	...
	3xx	1
	wc1n	1
	2667	1
	gsex	1
	chief	1


These are the most common words used in the emails within the whole dataset, however this not providing much help, so lets visualize the most common words per class (spam or ham).

.. code-block :: python	
	
	viz_data = spam_tokens.groupby('class')['word'].value_counts()
	viz_data = viz_data.reset_index(name = 'count')
	viz = viz_data.loc[viz_data['class'] == 'spam']
	viz1 = viz[0:10]
	sns.barplot(viz1, x = "word", y = 'count')
	plt.title("Most Common Words in Spam Emails")
	plt.xlabel("Word")
	plt.ylabel("Word Count")
	plt.show()


.. image:: .. image:: _build/html/_static/Tokenizationgraph1.png
   :alt: description
   :width: 400px

*graph will be embedded here*

Now we can analyze the top words in each class. The words in spam make sense with "free" or "claim", however, the top words in Ham make a little less sense. It almost seems like a lot of imformal text, which makes sense if emailing someone you know, but still pretty hard to interpet. This leads us to out topic of Relative Frequencies.






*Relative Frequency*
--------------------

# Relative Frequencies

While the processes of counting words in tokenization is useful, it sometimes can be hard to interpret. This is mainly due to documents containg hundred of thousands or even millions of tokens. So, in general, words tend to appear more, thus making their appeareance less meaningful. So we can then naturally go to use their text frequncy which can be defined as:

𝑇𝐹  = text frequency =  # of times 𝑤𝑜𝑟𝑑 appears in a document/total words in the document 

Furthermore, we can go on to define their relative frequencies which can be defined as:

						𝑅𝐹=relative frequency=𝑇𝐹 document 1/𝑇𝐹 document 2 

This now allows us to find which words are more frequnetly associated with each document.

Another addition to relative frequencies that assists in intepretation is the logratio. The logration is simply defined as:
𝑙𝑜𝑔(𝑇𝐹 document 1𝑇𝐹 document 2) 

This is important because if a word has a higher frequnecy in document 1, the logratio will be more positive, and thus more greatly associated with document 1. This is because  𝑙𝑜𝑔(𝐴𝐵)  = 𝑙𝑜𝑔(𝐴)−𝑙𝑜𝑔(𝐵) 

Example
^^^^^^^

**Calculating Relative Frequency**

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

*Our function is dividing the text frequency of a word in movies description divided by that same word in show descriptions. So by taking a logration of the relative frequency, we can see which word is more greatly associated with with category. Since we are dividing by the text frequency of show, and since  𝑙𝑜𝑔(𝐴/𝐵)=𝑙𝑜𝑔(𝐴)−𝑙𝑜𝑔(𝐵) , then a greater negative value means more greatly associated with show, and vice versa.*

.. code-block :: python

	mostfreq = pd.concat([  rel_freq[0:10] , rel_freq[-10:]  ])
	sns.barplot(data=mostfreq, x="logratio", y="word")
	plt.xlabel("Logratio")
	plt.show()

.. image:: /_static/Unknown.png
   :alt: Message class distribution
   :align: center
   :width: 400px





*Term-Frequency Inverse Document Frequency (TF-IDF)*
-------------------------------------------------

TF-IDF highlight terms are both frequent within a specifc document and unqiue across various documents.

Example
^^^^^^^

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

.. image:: _static/Unknown-2.png
   :alt: Message class distribution
   :align: center
   :width: 400px



