import coll_help
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
representer = coll_help.tf_idf(train_docs)
#print coll_help.feature_values(test_docs[0], representer)

def func1():
    alpha = 100
    betha = 20
    gamma = 50
    curTraining = 
    for t in range(0, betha): pass