import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import collections
import csv
import string
import re
from stemming.porter2 import stem

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import os.path
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "memo.csv")

stop = set(stopwords.words('english'))
stop_list=[]
for i in stop:
	if (nltk.pos_tag([i])[0][1]!='VBD' and nltk.pos_tag([i])[0][1]!='PRP$'):
		stop_list.append(i)
		
exclude = set(string.punctuation) 

def load_doc(filename):
    with open(filename) as csv_file:
        readCSV=csv.reader(csv_file.read().splitlines())
        sents_orig=[]
        for row in readCSV:
            x = row[0].lower()
            sents_orig.append(x)
    return sents_orig

def doc_cleaner(text):
    #tokenizes and stems the text
    re.sub(r'[^A-Za-z .-]+', ' ', text)
    stop_free = " ".join([i for i in text.split() if i not in stop_list])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(stem(word).decode("utf-8", 'ignore')
                          for word in punc_free.split())
    x=normalized.split()
    if(x[0].lower()=='remember'):
        x=x[1:]

    return x


def tfidf(sentences):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=doc_cleaner,
                                    max_df=0.96,
                                    min_df=0.04,
                                    lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    return tfidf_matrix, tfidf_vectorizer
        
def cluster_sentences(tfidf_matrix, nb_of_clusters):
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
    return kmeans.labels_, dict(clusters)


if __name__ == "__main__":
    sentences = load_doc(path)
    tfidf_matrix, tfidf = tfidf(sentences)
    all_clusters = []
    all_score = []
    all_sample_silhouette = []

    #set the range
    #
    for nclusters in range(3,7):
        cluster_labels, cluster = cluster_sentences(tfidf_matrix, nclusters)
        score = silhouette_score(tfidf_matrix, cluster_labels)
        sample_silhouette_values = silhouette_samples(tfidf_matrix, cluster_labels)
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
                    print "\tsentence ",i,": ",sentences[sentence]
