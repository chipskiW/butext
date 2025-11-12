
**Tokenization**
----------------

Tokenization is the process in which a particular portion of text is split into individual words or tokens. For example, the sentence "I like dogs" would become ["I" , "like", "dogs"]. While at its surface it seems like a pretty simple task, it assists with basic textual anaysis such as word counting and text frequencies all the way to Machine Learning, Natural Language Processing, and Deep Learning. In the 2017 paper "Attention Is All You Need" by Ashish Vaswani et al. which is the creation of the transfomer architecture which gave birth to the GPT decoder which is resposible for Chat-GPT, the architecture starts with tokenizing text and assign each word a vector embedding. Thus, while this task seems rudimentary, it has birthed many of the AI/Machine Learning developments in todays society.

Now we'll start with the basics of tokenization and see how it works:

**Example Code**

Importing Necessary Packages

.. code-block :: python

	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	import butext as bax



*Uploading Datset*

.. code-block :: python
	
	spam = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/HARP-210-NLP/main/datasets/SMSSpamCollection.tsv",sep="\t")
	spam.head(5)

*Output*

.. code-block :: none

		class	text
	0	ham	Go until jurong point, crazy.. Available only ...
	1	ham	Ok lar... Joking wif u oni...
	2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
	3	ham	U dun say so early hor... U c already then say...
	4	ham	Nah I don't think he goes to usf, he lives aro...


*Tokenization In Action*

Now that our data is loaded, we can use our tokenization function to find some basic information about our data. Our tokenization function is denoted by bax.tokenize and works best within pandas pipe() operator. We will also utilize our stop`words function denoted by bax.stopwords as it removes words not important to our analysis

.. code-block :: python

	spam_tokens = (
   	 	spam
    		.pipe(bax.tokenize, 'text')
    		.pipe(bax.stopwords, 'word')
		)
		spam_tokens

*Output*

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

*Output*

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

*Output*

.. image:: _build/html/_static/Tokenizationgraph1.png
	:alt: description
	:width: 400px

Now we can analyze the top words in each class. The words in spam make sense with "free" or "claim", however, the top words in Ham make a little less sense. It almost seems like a lot of imformal text, which makes sense if emailing someone you know, but still pretty hard to interpet. This leads us to out topic of Relative Frequencies.



**Sources**

*https://arxiv.org/pdf/1706.03762*


*https://www.geeksforgeeks.org/nlp/nlp-how-tokenizing-text-sentence-words-works/*


**Relative Frequency**
----------------------

While the processes of counting words in tokenization is useful, it sometimes can be hard to interpret. This is mainly due to documents containg hundred of thousands or even millions of tokens. So, in general, words tend to appear more, thus making their appeareance less meaningful. So we can then naturally go to use their text frequncy which can be defined as:

𝑇𝐹  = text frequency =  # of times 𝑤𝑜𝑟𝑑 appears in a document/total words in the document 

Furthermore, we can go on to define their relative frequencies which can be defined as:

						𝑅𝐹=relative frequency=𝑇𝐹 document 1/𝑇𝐹 document 2 

This now allows us to find which words are more frequnetly associated with each document.

Another addition to relative frequencies that assists in intepretation is the logratio. The logration is simply defined as:
𝑙𝑜𝑔(𝑇𝐹 document 1𝑇𝐹 document 2) 

This is important because if a word has a higher frequnecy in document 1, the logratio will be more positive, and thus more greatly associated with document 1. This is because  𝑙𝑜𝑔(𝐴𝐵)  = 𝑙𝑜𝑔(𝐴)−𝑙𝑜𝑔(𝐵) 

*Example Code*

**Necessary Imports**

.. code-block :: python

	import butext as bax
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns

**Load Data** 

spam = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/HARP-210-NLP/main/datasets/SMSSpamCollection.tsv", sep="\t")
spam.head(5)


**Relative Frequency In Action**

.. code-block :: python

	rel_freq = (
    		spam
    		.pipe(bax.tokenize, 'text') #tokenize text
    		.pipe(bax.stopwords, 'word') # removes stopwords
    		.groupby('class')['word'].value_counts(normalize = True) # calculates text frequencies per class
    		.reset_index()
    		.query('proportion > 0.0005') # removes meaningless words
    		.pipe(bax.rel_freq, 'class') # calculates relative frequency
	)

.. code-block :: python
	
	rel_freq = rel_freq.sort_values(by = 'logratio', ascending= True)
	rel_freq

**Output**

.. code-block :: none 

		class	word	ham	spam	rel_freq	logratio
	597	txt	0.000250	0.012698	0.019688	-1.705791
	391	mobile	0.000250	0.010412	0.024010	-1.619605
	145	claim	0.000250	0.009566	0.026135	-1.582778
	462	prize	0.000250	0.007788	0.032101	-1.493488
	653	won	0.000250	0.006180	0.040455	-1.393023
	...	...	...	...	...	...
	681	ü	0.004536	0.000250	18.142780	1.258704
	152	come	0.006173	0.000250	24.691358	1.392545
	358	lt	0.007488	0.000250	29.951691	1.476421
	423	ok	0.007488	0.000250	29.951691	1.476421
	266	gt	0.007622	0.000250	30.488459	1.484135


.. code-block :: python

	mostfreq = pd.concat([  rel_freq[0:10] , rel_freq[-10:]  ])
	sns.barplot(data=mostfreq, x="logratio", y="word")
	plt.xlabel("Logratio")
	plt.show() 

**Output**

.. image:: _build/html/_static/relativefrequencygraph1.png
	:alt: description
	:width: 400px

This graph visualize the top 10 words most associated with spam and ham emails, with spam being positive and ham being negative. We can see we get some of the same words from the general word counting, but also get some new ones. These words are slightly more interperetable than with word counting. However there is even a better measure for this, and it is called tf-idf.


**Term-Frequency Inverse Document Frequency (TF-IDF)**
------------------------------------------------------

TF-IDF stands for text-frequency inverse document-frequency and it used to study mutliple texts. This improves on relative frequency as it can only work on two. TF-IDF is defined as:

𝑇𝐹⋅𝐼𝐷𝐹=𝑇𝐹ln(1𝐷𝐹)=−𝑇𝐹ln(𝐷𝐹) 

Where TF is text frequency:
𝑇𝐹=text frequency=# of times 𝑤𝑜𝑟𝑑 appears in a documenttotal words in the document 
and DF is document frequency:
𝐷𝐹=document frequency=# of documents 𝑤𝑜𝑟𝑑 appears in# of documents 

This improves upon basic word counting and relative frequecies as it measures the uniquness of a word to a given document, while relative frequency and word counting does not. The logarithm is good because  𝑙𝑜𝑔(1)=0 , so any word that appears in every document will have a TF-IDF of zero and is not unique. This process helps naturally remove stopwords, however, it might be still practice to remove them manually.

TF-IDF can also be used to help create machine learning models like Support Vector Machines and Logistic Regressions as it turns words is specific numerical values (or vectors). This will explored later.

**Example Code**


Now that our data is loaded, we can use our TF-IDF function to find the most unique word to each class, which is spam or ham. We will use bax.tf_idf and is best used with pipe() operator. We will also utilize our stopewords function denoted by bax.stopwords as it removes words not important to our analysis. However, this is not super important for TF-IDF as it drives many stopwords to a zero value naturally.

.. code-block :: python

	spam_tfidf = (
    		spam
    		.pipe(bax.tokenize, 'text') # tokenizes text
    		.pipe(bax.stopwords, 'word') # removes stopwords
    		.groupby('class')['word']
    		.value_counts(normalize=True) # text frequency
    		.reset_index()
    		.pipe(bax.tf_idf, col='class') # tf-idf calculation
	)
	x = spam_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0] # many words will have tf_idf = 0 but those words aren't 		important, so we can filter them out for cleaner results
	x


.. code-block :: python
	
	graph = x[0:10]
	sns.barplot(graph, x = "word", y = 'tf_idf', hue = "class" , legend= True)
	plt.title('Most Unique Words per Class of Email')
	plt.xlabel('TF-IDF')
	plt.ylabel('Word')
	plt.xticks(rotation = 45)
	plt.show()

**Output** 

.. image:: _build/html/_static/TF-IDFgraph.png
	:alt: description
	:width: 400px 

Here we can see which words are the most unique to the spam and ham classes. Now we can finally see which words are truy most associated with each class of email. There is some of the same words from relative frequencies and word counting and there is some new ones. Which one you choose depends on the context of the situation. In situations where there are more than two documents, you need to use TF-IDF instead of relative frequencies. When there is only two documents, relative frequencies is still a great method, and it is a bit more interpretable as well. In the case of Machine Learning, TF-IDF is definitely the better measure to proceed with. Overall, they are both great methods for textual anlysis depending on the context, and most importantly is all possible due to tokenization.




