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



Term Frequency Inverse Document Frequency (TF-IDF)
-------------------------------------------------

TF-IDF Example
====================

.. code-block:: python

   text = "Read the Docs makes documentation easy."
    tokens = (
    df
    .pipe(bax.tokenize,'text')
    )
    tokens.head()
 

.. code-block:: none

   ['read', 'the', 'docs', 'makes', 'documentation', 'easy']
