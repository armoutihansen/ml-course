# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# import data
df = pd.read_csv('data/Lab7_TextAnalysisChristmasSongsFull.csv')
df.head()
df_text = df[['text']]
df_text['index'] = df_text.index
stop_words = stopwords.words('english')
documents = df_text


def lemmatize_stemming(text):
    return PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = ''
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result +=' '+ (lemmatize_stemming(token))
    return result

processed_docs = documents['text'].map(preprocess)
processed_docs[:10]
train_doc, test_doc, train_y, test_y = train_test_split(processed_docs, df['Popular'], random_state=181, test_size=0.25)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 3))
X = vectorizer.fit_transform(processed_docs)
dense = X.todense()
denselist = dense.tolist()
feature_names = vectorizer.get_feature_names()
pd.DataFrame(denselist, columns=feature_names)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation()
lda.fit(X)
lda.transform(X[331])
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

tf_feature_names = vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, 20)
X_new = lda.fit_transform(X)
from sklearn.linear_model import LogisticRegressionCV
logreg = LogisticRegressionCV(cv=5)
logreg.fit(X,df['Popular'])
logreg.score(X,df['Popular'])

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
params = {'C': np.logspace(0, 2)}
cv = GridSearchCV(LogisticRegressionCV(),params, cv=5).fit(X, df['Popular'])
cv.score(X, df['Popular'])
cv.best_estimator_
