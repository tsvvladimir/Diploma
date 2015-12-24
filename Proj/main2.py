import coll_help
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from mlxtend.preprocessing import DenseTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('clf', OneVsRestClassifier(LinearSVC())),
    ('clf', DecisionTreeClassifier()),
])

#test_id = test_id[:10]

def func1():
    alpha = 100
    betha = 200
    gamma = 50
    curTraining = train_id[:betha]
    #test_id = curTraining
    unlabeled = train_id[betha:]
    #for t in range(0, betha):
        #representer = coll_help.tf_idf(train_docs)
    mb = MultiLabelBinarizer()
    #print "labels:", [tuple(id_cat[id]) for id in curTraining]
    pipeline.fit([coll_help.reuters.raw(id) for id in curTraining], mb.fit_transform([id_cat[id] for id in curTraining]))
    predicted = pipeline.predict([coll_help.reuters.raw(id) for id in test_id])
    #print "predicted", predicted
    print np.array(test_id)
    pred =  np.array(mb.inverse_transform(predicted))
    print "inverse predicted", pred
    #print "classes", mb.classes_
    mb1 = MultiLabelBinarizer()
    t_real = mb1.fit_transform([id_cat[id] for id in test_id])
    t_pred = mb1.transform(pred)
    print "f1 score", f1_score(t_real, t_pred, average='micro')
    # mb.fit_transform([id_cat[id] for id in test_id])
    #print f1_score(mb.fit_transform([id_cat[id] for id in test_id]), predicted, average='macro')
    #print pipeline.score(mb.fit_transform([id_cat[id] for id in test_id]), predicted)

#print f1_score([(1, 2)], [(1)], average='samples')
func1()