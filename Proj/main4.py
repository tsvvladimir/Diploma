from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'sci.space',
]

print("Loading 20 newsgroups dataset for categories:")
print(categories)

data = fetch_20newsgroups(subset='train', categories=categories)
test_data = fetch_20newsgroups(subset='test', categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC())),
])

pipeline.fit(data.data, data.target)
predicted = pipeline.predict(test_data.data)
#print predicted
#print test_data.target

t = {}
t['0'] = 0
t['1'] = 0
for pred, exact in zip(predicted, test_data.target):
    if pred != exact:
        t['0'] += 1
    else:
        t['1'] += 1
print t