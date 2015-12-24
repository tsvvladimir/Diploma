import coll_help
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import active_algo
# coll_help.collection_stats()
# print coll_help.cachedStopWords
train_docs = []
test_docs = []
#print coll_help.reuters.categories(coll_help.reuters.fileids()[0])
fileid_cat = {}
for doc_id in coll_help.reuters.fileids():
    fileid_cat[doc_id] = coll_help.reuters.categories(doc_id)
    #if len(fileid_cat[doc_id]) > 1:
    #    print doc_id, "more than 1 class"
    if doc_id.startswith("train"):
        train_docs.append(coll_help.reuters.raw(doc_id))
    else:
        test_docs.append(coll_help.reuters.raw(doc_id))
print "training set", len(train_docs)
print "testing set", len(test_docs)
#print fileid_cat
#representer = coll_help.tf_idf(train_docs)
#print coll_help.feature_values(test_docs[0], representer)

#classifier = Pipeline([
#        ('tfidf', TfidfVectorizer(tokenizer=coll_help.tokenize, min_df=3, max_df=0.90, max_features=3000, use_idf=True, sublinear_tf=True, norm='l2')),
#        ('clf', OneVsRestClassifier(LinearSVC()))])

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('to_dense', DenseTransformer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

def func1():
    alpha = 100
    betha = 20
    gamma = 50
    curTraining = train_docs[:betha]
    unlabeled = train_docs[betha:]
    #for t in range(0, betha):
        #representer = coll_help.tf_idf(train_docs)
    classifier.fit(curTraining, [fileid_cat[doc_id] for doc_id in curTraining])
    predicted = classifier.predict(test_docs)
    for item, labels in zip(X_test, predicted):
        print "predicted:", labels#, "real:", fileid_cat[item]

func1()
