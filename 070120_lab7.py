import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim
from tokenizer_xm import text_tokenizer_xm

df = pd.read_csv('data/Lab7_TextAnalysisChristmasSongsFull.csv')

X_train, X_test, y_train, y_test = train_test_split(df[['text']], df['Popular'], test_size = 0.25,random_state = 23)

X_train.reset_index(drop = True,inplace = True)
X_test.reset_index(drop = True,inplace = True)
y_train.reset_index(drop = True,inplace = True)
y_test.reset_index(drop = True,inplace = True)

def lemmatize_stemming(text):
    return PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


docs = df['text'].map(text_tokenizer_xm)
docs.txt_pre_pros()

for i in docs:
    print(i.txt_pre_pros())
vec_tfidf = TfidfVectorizer(ngram_range =(1,1),min_df = 0.1, max_df = 1.0)
vec_tfidf_f = vec_tfidf.fit(X_train['text'])
X_train['text']
train_dtm_ngram = vec_tfidf_f.transform(X_train)
dense = train_dtm_ngram.todense()
denselist = dense.tolist()
feature_names = vec_tfidf_f.get_feature_names()
pd.DataFrame(denselist, columns=feature_names)
train_dtm_ngram.shape
list(gensim.parsing.preprocessing.STOPWORDS)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train['text'])
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X_train_tfidf.toarray(), y_train)
clf.score(X_train_tfidf.toarray(), y_train)

X_test_counts = count_vect.transform(X_test['text'])
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
clf.score(X_test_tfidf.toarray(), y_test)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 20)
lda_f = lda.fit(X_train_counts)
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

tf_feature_names = count_vect.get_feature_names()
print_top_words(lda_f, tf_feature_names, 20)
lda_weights = lda_f.transform(X_train_counts)
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=200)
xgbc_lda = xgbc.fit(lda_weights,y_train)
