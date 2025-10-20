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

.. code-block:: python

   text = 'Welcome to our Butext documentation!'
    tokens = (
    df
    .pipe(bax.tokenize,'text')
    )
    tokens.head()

Output

.. code-block:: none

   ['welcome', 'to', 'our', 'butext', 'documentation']

Relative Frequency 
------------------

*Relative Frequency is a simple mathematical operation that divides text frequency of a single word from one document by the text frequency of the same word from a different document*

Relative Frequency Example
====================

.. code-block:: python

    df_tfidf = (
    df
    .groupby('doc')['text']
    .value_counts(normalize=True)
    .reset_index()
    .pipe(bax.tf_idf, col='text')

Output
 

.. code-block:: none

   ['read', 'the', 'docs', 'makes', 'documentation', 'easy']



Term-Frequency Inverse Document Frequency (TF-IDF)
-------------------------------------------------

TF-IDF allows us to measure the uniqueness of a word to a given document.

TF-IDF Example
====================

.. code-block:: python


.. code-block:: none

   ['read', 'the', 'docs', 'makes', 'documentation', 'easy']
