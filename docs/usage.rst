=======
Usage
=======



Here is an example of importing the proper libraries and functions:

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
