# imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
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

# import data
df = pd.read_csv('data/Lab7_TextAnalysisChristmasSongsFull.csv')

# functions that lemmatize, stem and preprocess text


def lemmatize_stemming(text):
    return PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = ''
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result += ' ' + (lemmatize_stemming(token))
    return result


# preparing documents
docs = df['text'].map(preprocess)

# splitting data into train and test set
train_docs, test_docs, train_y, test_y = train_test_split(
    docs, df['Popular'], random_state=181, test_size=0.25)

tfidf = TfidfVectorizer()
lda = LatentDirichletAllocation()
logistic = LogisticRegression()

params = {'tfidf__ngram_range': [(1,1),(1,2),(2,2)], 'lda__n_components': np.arange(5,30), 'logistic__C':np.logspace(-4,4,20)}

pipe = Pipeline(steps=[('tfidf', tfidf), ('lda', lda), ('logistic', logistic)])

cv = GridSearchCV(pipe, params, cv=5, verbose=1, n_jobs=-1)
cv.cv_results_
cv.fit(train_docs, train_y)
cv.best_estimator_
cv.score(train_docs, train_y)
cv.score(test_docs, test_y)
cv.get_params()
cv.cv_results_
X_train = tfidf.fit_transform(train_docs)
lr = LogisticRegressionCV(cv=5).fit(X_train, train_y)
lr.scores_
lr.score(X_train, train_y)
X_test = tfidf.transform(test_docs)
lr.score(X_test, test_y)
lr.C_
np.logspace(-4,4,20)
np.argmax(cv.cv_results_['mean_test_score'])
cv.cv_results_['params'][np.argmax(cv.cv_results_['mean_test_score'])]
cv.cv_results_['mean_test_score']
cv.best_estimator_
from sklearn.pipeline import make_pipeline
def myRegression(ngrams=(1,1), comp=10):
    return make_pipeline(TfidfVectorizer(ngram_range=ngrams), LatentDirichletAllocation(n_components=comp), LogisticRegression())
param_grid = {'myRegression__ngrams': [(1,1),(1,2),(1,3)]}
grid = GridSearchCV(myRegression(), param_grid, cv=5)
grid.fit(train_docs, train_y)



params2 = {'tfidf__ngram_range': [(1,1),(1,2)], 'logistic__C':np.logspace(-4,4,20)}

pipe2 = Pipeline(steps=[('tfidf', tfidf), ('logistic', logistic)])

cv2 = GridSearchCV(pipe2, params2, cv=5, verbose=1, n_jobs=-1)

cv2.fit(train_docs, train_y).decision_function(train_docs)
cv2.score(train_docs, train_y)
cv2.score(test_docs, test_y)
np.argmax(cv2.cv_results_['mean_test_score'])
cv2.cv_results_['params'][np.argmax(cv2.cv_results_['mean_test_score'])]

from sklearn.metrics import classification_report
pred = cv2.predict(test_docs)
print(classification_report(test_y, pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
roc_auc_score(test_y, pred)

y_score = cv2.decision_function(test_docs)

cv2.decision_function(test_docs)

fpr, tpr, _ = roc_curve(test_y, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='orange',
         label='ROC curve (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for our LDA')
plt.legend(loc="lower right")
from sklearn.metrics import confusion_matrix

print(pd.DataFrame(confusion_matrix(test_y, pred).T,
                                    index = ['No', 'Yes'], columns = ['No', 'Yes']))

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent').fit(train_docs, train_y)
dummy.score(train_docs, train_y)
dummy.score(test_docs, test_y)
predd = dummy.predict(test_docs)
predd
np.mean(test_y)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000)


params3 = {'tfidf__ngram_range': [(1,1),(1,2)], 'rf__max_features': ['auto', 'log2', None, 1]}

pipe3 = Pipeline(steps=[('tfidf', tfidf), ('rf', rf)])

cv3 = GridSearchCV(pipe3, params3, cv=5, verbose=1, n_jobs=-1)

cv3.fit(train_docs, train_y)
cv3.score(train_docs, train_y)
cv3.score(test_docs, test_y)
np.argmax(cv3.cv_results_['mean_test_score'])
cv3.cv_results_['params'][np.argmax(cv3.cv_results_['mean_test_score'])]
np.max(cv3.cv_results_['mean_test_score'])

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

params4 = {'tfidf__ngram_range': [(1,1),(1,2)], 'knn__n_neighbors': [1,2,3,4,5,6,7,8,9,10]}

pipe4 = Pipeline(steps=[('tfidf', tfidf), ('knn', knn)])

cv4 = GridSearchCV(pipe4, params4, cv=5, verbose=1, n_jobs=-1)

cv4.fit(train_docs, train_y)

cv4.score(train_docs, train_y)
cv4.score(test_docs, test_y)
np.argmax(cv4.cv_results_['mean_test_score'])
cv4.cv_results_['params'][np.argmax(cv4.cv_results_['mean_test_score'])]
np.max(cv4.cv_results_['mean_test_score'])
cv4.best_estimator_

from sklearn.ensemble import GradientBoostingClassifier
original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None,'min_samples_split': 5}
clf = GradientBoostingClassifier()
params5 = {'tfidf__ngram_range': [(1,1),(1,2)], 'clf__learning_rate':np.linspace(0.01, 0.9, 5), 'clf__n_estimators':list(range(100, 1001, 200)), 'clf__subsample':np.linspace(0.1, 1, 5)}
np.linspace(0.1, 1, 10)
pipe5 = Pipeline(steps=[('tfidf', tfidf), ('clf', clf)])
list(range(100, 1001, 100))
cv5 = GridSearchCV(pipe5, params5, cv=5, verbose=1, n_jobs=-1)

cv5.fit(train_docs, train_y)
cv5.score(train_docs, train_y)
cv5.score(test_docs, test_y)
np.argmax(cv5.cv_results_['mean_test_score'])
cv5.cv_results_['params'][np.argmax(cv5.cv_results_['mean_test_score'])]
np.max(cv5.cv_results_['mean_test_score'])
cv5.best_estimator_
