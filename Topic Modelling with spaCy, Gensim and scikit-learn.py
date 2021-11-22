#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language # To use during adding component to our spaCy pipeline

from tqdm import tqdm_notebook as tqdm # Module to see the progress bars
from pprint import pprint # For a better print font


# In[ ]:


nlp = spacy.load("en_core_web_sm")


# In[ ]:


paper = pd.read_csv(r"C:\Users\User\Downloads\papers.csv")
paper = paper.sample(1000)


# In[ ]:


paper.head()


# In[ ]:


paper = paper.drop(columns = ['source_id','year','title','abstract'])


# In[ ]:


full_text = paper['full_text']


# In[ ]:


# Adding additional specific stop words after looking at the data beforehand. 
# These stopwords include commonly used notations in the papers.
stop_list = [" ","'s","D.","\n","\n\n","=","ef","0","1","2","3","4","5","6","7","8","9","10" 
             "a","b","c","d","e","f","g","j","k","l","m","n","o","p","r","s","t","u","v","y","z",
             "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","Y","Z"
            "✓","−","∈"]

# Updating spaCy's default stop words list. 
nlp.Defaults.stop_words.update(stop_list)

# Iterates over the words in the stop words list and resets the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True


# In[ ]:


# Since this is the third version (v3) of spaCy, @Language.component("function_name") should be used first.
# The difference of this version from the previous ones is the need for using a string when the nlp.add_pipe() is used.
# Rather than a direct function name.

@Language.component("remove_stopwords")  
def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim and scikit-learn.
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc

# The add_pipe function appends our functions to the default pipeline.
nlp.add_pipe('remove_stopwords')


# In[ ]:


# Making an empty list for Gensim model. Then each article is iterated the result of which is to be appended to this empty list.
doc_list = []
for doc in tqdm(full_text):
    pr = nlp(doc)
    doc_list.append(pr)


# In[ ]:


doc_list[3]


# In[ ]:


#----------------------------------------------
print("-"*100)
print("Doing Topic Modelling with Gensim")


# In[ ]:


# Creating, which is a mapping of word IDs to words.
words = corpora.Dictionary(doc_list)

# Turning each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in doc_list]


# In[ ]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=words,
                                           num_topics=10, 
                                           random_state=2,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


pprint(lda_model.print_topics(num_words=5))


# In[ ]:


graph = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary = lda_model.id2word)
graph


# In[ ]:


#-------------------------------------------------------------------------------------------
print("-"*100)
print("Doing Topic Modelling with scikit-learn")


# In[ ]:


# Creating TF vectorizer for LDA of scikit-learn to use.
# LDA uses TF Vectorizer rather than TF-IDF since it is a probabilistic model.
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b',
                                max_df = 0.5, 
                                min_df = 10)

dtm_tf = tf_vectorizer.fit_transform(pr)


# In[ ]:


# Forming our LDA model.
lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
lda_tf.fit(dtm_tf)


# In[ ]:


# Using pyLDAvis for visualization.
graph_2 = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
graph_2


# In[ ]:




