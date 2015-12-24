import coll_help
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle
import copy
from numpy.random import choice

train_id = []
test_id = []

id_cat = {}

for id in coll_help.reuters.fileids():
    id_cat[id] = coll_help.reuters.categories(id)
    if id.startswith("train"):
        train_id.append(id)
    else:
        test_id.append(id)

print len(id_cat)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
        #('clf', DecisionTreeClassifier()),])
    ('clf', OneVsRestClassifier(LinearSVC()))])


def find_classifier():
    alpha = 100 #initial training set
    betha = 20 #number of iterations
    gamma = 50 #number of sampling
    curTraining = train_id[:alpha]
    unlabeled = train_id[alpha:]

    scores = []
    #fit classifier

    mb = MultiLabelBinarizer()
    pipeline.fit([coll_help.reuters.raw(id) for id in curTraining], mb.fit_transform([id_cat[id] for id in curTraining]))
    predicted = pipeline.predict([coll_help.reuters.raw(id) for id in test_id])
    pred =  np.array(mb.inverse_transform(predicted))
    mb1 = MultiLabelBinarizer()
    t_real = mb1.fit_transform([id_cat[id] for id in test_id])
    t_pred = mb1.transform(pred)
    score = f1_score(t_real, t_pred, average='micro')
    scores.append(score)
    final_score = sum(scores)/len(scores)
    print "f1 final score train set", final_score, "train_set length", len(curTraining)

    for t in range(1, betha):
        scores = []
        #add texts to training set

        #res

        #print "score for zero unlabeled", pipeline.score(coll_help.reuters.raw(unlabeled[0]), id_cat[unlabeled[0]])

        res = np.random.randint(0, len(unlabeled), gamma)

        #print "gamma was:", gamma, "choice len", len(res)
        unlabeled_copy = list(unlabeled)
        for i in res:
            try:
                curTraining.append(unlabeled[i])
                unlabeled_copy.pop(i)
            except:
                print "oops!", i, "unlabeled length", len(unlabeled)
        unlabeled = unlabeled_copy
        #curTraining = list(set(curTraining) | set(res))
        #unlabeled = list(set(unlabeled) - set(res))


        #fit classifier

        mb = MultiLabelBinarizer()
        pipeline.fit([coll_help.reuters.raw(id) for id in curTraining], mb.fit_transform([id_cat[id] for id in curTraining]))
        predicted = pipeline.predict([coll_help.reuters.raw(id) for id in test_id])
        pred =  np.array(mb.inverse_transform(predicted))
        mb1 = MultiLabelBinarizer()
        t_real = mb1.fit_transform([id_cat[id] for id in test_id])
        t_pred = mb1.transform(pred)
        score = f1_score(t_real, t_pred, average='micro')
        scores.append(score)
        final_score = sum(scores)/len(scores)
        print "f1 final score train set", final_score, "train_set length", len(curTraining)


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('clf', DecisionTreeClassifier()),])
    ('clf', OneVsRestClassifier(LinearSVC()))])

def find_classifier1():
    alpha = 100 #initial training set
    betha = 20 #number of iterations
    gamma = 50 #number of sampling
    curTraining = train_id[:alpha]
    unlabeled = train_id[alpha:]

    scores = []
    #fit classifier

    mb = MultiLabelBinarizer()
    pipeline.fit([coll_help.reuters.raw(id) for id in curTraining], mb.fit_transform([id_cat[id] for id in curTraining]))
    predicted = pipeline.predict([coll_help.reuters.raw(id) for id in test_id])
    pred =  np.array(mb.inverse_transform(predicted))
    mb1 = MultiLabelBinarizer()
    t_real = mb1.fit_transform([id_cat[id] for id in test_id])
    t_pred = mb1.transform(pred)
    score = f1_score(t_real, t_pred, average='micro')
    scores.append(score)
    final_score = sum(scores)/len(scores)
    print "f1 final score train set", final_score, "train_set length", len(curTraining)

    for t in range(1, betha):
        scores = []
        #add texts to training set

        #res

        #print "score for zero unlabeled", pipeline.score(coll_help.reuters.raw(unlabeled[0]), id_cat[unlabeled[0]])

        #res = np.random.randint(0, len(unlabeled), gamma)

        #print "gamma was:", gamma, "choice len", len(res)
        #unlabeled_copy = list(unlabeled)
        #for i in res:
        #    try:
        #        curTraining.append(unlabeled[i])
        #        unlabeled_copy.pop(i)
        #    except:
        #        print "oops!", i, "unlabeled length", len(unlabeled)
        #unlabeled = unlabeled_copy
        #curTraining = list(set(curTraining) | set(res))
        #unlabeled = list(set(unlabeled) - set(res))

        n_clusters = 8
        pipelinecluster = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clusterer', KMeans(n_clusters=n_clusters)),])

        labels = pipelinecluster.fit_predict([coll_help.reuters.raw(id) for id in unlabeled])
        print labels
        c = Counter(labels)
        for cluster_number in range(n_clusters):
            print "cluster", cluster_number, "num elem in cluster", c[cluster_number]
        print "cluster centers", pipelinecluster.named_steps['clusterer'].cluester_centers_


        #fit classifier

        mb = MultiLabelBinarizer()
        pipeline.fit([coll_help.reuters.raw(id) for id in curTraining], mb.fit_transform([id_cat[id] for id in curTraining]))
        predicted = pipeline.predict([coll_help.reuters.raw(id) for id in test_id])
        pred =  np.array(mb.inverse_transform(predicted))
        mb1 = MultiLabelBinarizer()
        t_real = mb1.fit_transform([id_cat[id] for id in test_id])
        t_pred = mb1.transform(pred)
        score = f1_score(t_real, t_pred, average='micro')
        scores.append(score)
        final_score = sum(scores)/len(scores)
        print "f1 final score train set", final_score, "train_set length", len(curTraining)



#count f1 score using all training set as baseline
print "start scoring without active"
'''
scores = []
print "current iter", i
mb = MultiLabelBinarizer()
pipeline.fit([coll_help.reuters.raw(id) for id in train_id], mb.fit_transform([id_cat[id] for id in train_id]))
predicted = pipeline.predict([coll_help.reuters.raw(id) for id in test_id])
pred =  np.array(mb.inverse_transform(predicted))
mb1 = MultiLabelBinarizer()
t_real = mb1.fit_transform([id_cat[id] for id in test_id])
t_pred = mb1.transform(pred)
score = f1_score(t_real, t_pred, average='micro')
scores.append(score)
final_score = sum(scores)/len(scores)
print "f1 final score without active", final_score
'''
print "f1 final score without active", 0.861722278887, "train_set length", len(train_id)

print "start fitting random sampling"

find_classifier()

print "start fitting cluster sampling"

find_classifier1()