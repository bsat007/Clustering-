from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

import csv
import collections
import string
from stemming.porter2 import stem
import re

import numpy as np

import nltk
from nltk.corpus import stopwords 

import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import os.path
my_path = os.path.abspath(os.path.dirname(__file__))

#Set the path --- 
#
path = os.path.join(my_path, "memo.csv")

##stopwords
stop = set(stopwords.words('english'))
stop_list=[]
for i in stop:
	if (nltk.pos_tag([i])[0][1]!='VBD' and nltk.pos_tag([i])[0][1]!='PRP$'):
		stop_list.append(i)
		
exclude = set(string.punctuation) 

def load_doc_clean(filename):
    with open(filename) as csv_file:
        readCSV=csv.reader(csv_file.read().splitlines())
        sents=[]
        sents_orig=[]
        for row in readCSV:
            x = row[0].lower()
            sents_orig.append(x)
            re.sub(r'[^A-Za-z .-]+', ' ', x)
            stop_free = " ".join([i for i in x.split() if i not in stop_list])
            punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
            normalized = " ".join(stem(word).decode("ISO-8859-1", 'ignore')
                                  for word in punc_free.split())
            x=normalized.split()
            if(x[0].lower()=='remember'):
                x=x[1:]
            normalized = ' '.join(x)
            sents.append(normalized)
    return sents,sents_orig

#load and clean the document   
doc_complete, orig_doc = load_doc_clean(path)

sentences=[]
for i, line in enumerate(doc_complete):
    sentences.append(TaggedDocument(gensim.utils.to_unicode(line).split(),
                                    [i]))

word_dict = []
for doc in doc_complete:
    for token in doc.split():
        word_dict.append(token)

word_dict = nltk.FreqDist(word_dict)

#Train the model
"""
min_count = ignore all words with total frequency lower than this.

iter = number of iterations (epochs) over the corpus.
        The default inherited from Word2Vec is 5.

workers = use this many worker threads to train the model.

window is the maximum distance between the predicted word and
        context words used for prediction within a document.

size is the dimensionality of the feature vectors.
"""
model = gensim.models.doc2vec.Doc2Vec(size=30, window=8, iter=4000,
                                      workers=10, min_count=2)

model.build_vocab(sentences)

model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

#Get the feature vector for each sentence
#
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0
    index2word_set = set(model.wv.index2word)
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            x = model[word]*(word_dict[word]/len(word_dict))
            featureVec = np.add(featureVec, x)
    if(nwords!=0):
            featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(sentences, model, num_features):
    # Given a set of sentences (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0

    sentenceFeatureVecs = np.zeros((len(sentences),num_features),
                                   dtype="float32")
    
    for sentence in sentences: 
       sentenceFeatureVecs[counter] = makeFeatureVec(sentence.words,
                                                     model, num_features)
       counter+=1
       
    return sentenceFeatureVecs


def cluster_sentences(sentences, nb_of_clusters, doc2vec_matrix):
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(doc2vec_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return kmeans.labels_, dict(clusters)


if __name__ == "__main__":

    num_features = model.vector_size
    #Obtain Doc2vec matrix
    #
    doc2vec_matrix = getAvgFeatureVecs(sentences, model,
                                       num_features)
    all_clusters = []
    all_score = []
    all_sample_silhouette = []

    #set the range
    #
    for nclusters in range(3,7):
        cluster_labels, cluster = cluster_sentences(sentences, nclusters,
                                                    doc2vec_matrix)

        score = silhouette_score(doc2vec_matrix, cluster_labels)

        sample_silhouette_values = silhouette_samples(doc2vec_matrix,
                                                      cluster_labels)
        all_clusters.append(cluster)
        all_score.append(score)
        all_sample_silhouette.append(sample_silhouette_values)

    #get the clusters for which silhouette_score is maximum
    #
    clusters = all_clusters[all_score.index(max(all_score))]
    nclusters = all_score.index(max(all_score))+2

    #print the cluster
    #
    for cluster in range(nclusters):
            print "CLUSTER",cluster,":"
            for i,sentence in enumerate(clusters[cluster]):
                    print "\tsentence ",i,": ",orig_doc[sentence]
