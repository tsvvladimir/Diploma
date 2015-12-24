import coll_help
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from mlxtend.preprocessing import DenseTransformer
from sklearn.preprocessing import MultiLabelBinarizer

train_id = []
test_id = []

id_cat = {}

for id in coll_help.reuters.fileids():
    id_cat[id] = coll_help.reuters.categories(id)
    if id.startswith("train"):
        train_id.append(id)
    else:
        test_id.append(id)

#classifier = Pipeline(
#    [
#        ('tfidf', TfidfVectorizer(tokenizer=coll_help.tokenize, min_df=3, max_df=0.90, max_features=3000, use_idf=True, sublinear_tf=True, norm='l2')),
#        ('clf', OneVsRestClassifier(LinearSVC()))
#    ]
#)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('to_dense', DenseTransformer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

def func1():
    alpha = 100
    betha = 20
    gamma = 50
    curTraining = train_id[:betha]
    unlabeled = train_id[betha:]
    #for t in range(0, betha):
        #representer = coll_help.tf_idf(train_docs)
    classifier.fit([coll_help.reuters.raw(id) for id in curTraining], MultiLabelBinarizer().fit_transform([id_cat[id] for id in curTraining]))
    predicted = classifier.predict([coll_help.reuters.raw(id) for id in test_id])
    print predicted

func1()