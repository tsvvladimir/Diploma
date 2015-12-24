import coll_help
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier()),])

def find_classifier():
    alpha = 100
    betha = 20
    gamma = 50
    curTraining = train_id[:alpha]
    unlabeled = train_id[alpha:]

    scores = []
    #fit classifier
    for i in range(0, 5):
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
    print "f1 final score train set", final_score

    for t in range(1, betha):
        scores = []
        #add texts to training set

        #res
        res = choice(unlabeled, gamma)
        #ranking = map(lambda id: pipeline.predict(coll_help.reuters.raw(id)), unlabeled)
        #for id in unlabeled:
        #    rank = pipeline.predict(coll_help.reuters.raw(id))
        #    print "rank", mb.inverse_transform(rank)
        #print "ranking", ranking
        curTraining = list(set(curTraining) | set(res))
        unlabeled = list(set(unlabeled) - set(res))

        #fit classifier
        for i in range(0, 5):
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
        print "f1 final score train set", final_score


#count f1 score using all training set as baseline
scores = []
for i in range(0, 5):
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

find_classifier()