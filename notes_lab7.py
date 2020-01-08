import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv('data/Lab7_TextAnalysisChristmasSongsFull.csv')
df.head()
df.Popular.mean()
df.text.head()
df_text = df[['text']]
df_text['index'] = df_text.index
df_text.head()
stop_words = stopwords.words('english')

documents = df_text

print(len(documents))
print(documents[:5])

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
nltk.download('wordnet')
stemmer = PorterStemmer()
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 431].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['text'].map(preprocess)
processed_docs[:10]
len(dictionary)
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
secret = 0
for k, v in dictionary.iteritems():
    if 'secret' in v:
        secret += 1

secret


    count += 1
    if count > 10:
        break
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[431]

bow_doc_431 = bow_corpus[431]
for i in range(len(bow_doc_431)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_431[i][0],
                                               dictionary[bow_doc_431[i][0]],
bow_doc_431[i][1]))


preprocess(doc_sample)


from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
corpus_tfidf
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

for index, score in sorted(lda_model_tfidf[bow_corpus[331]], key=lambda tup: -1*tup[1]):
    print("Score: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

lda_model_tfidf[bow_corpus[431]]
lda_model_tfidf.get_document_topics(bow_corpus[431], minimum_probability=0.0)
vec = []
for i in range(len(bow_corpus)):
    topics = lda_model_tfidf.get_document_topics(bow_corpus[i], minimum_probability=0.0)
    topics = [topics[j][1] for j in range(len(topics))]
    vec.append(topics)
np.array(vec)
lda_model_tfidf.print_topics()
lda_model_tfidf[bow_corpus[331]]

from sklearn.linear_model import LogisticRegressionCV
log = LogisticRegressionCV().fit(vec, df['Popular'])
log.scores_
log.score(vec, df['Popular'])
log.Cs_
1e-4
len(np.arange(0.0001,10000))
np.exp(0.0001)
np.exp(10000)
np.logspace(-4,4)
list(corpus_tfidf)
