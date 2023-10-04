import utility as util
import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import similarities
from matplotlib import pyplot as plt
import spacy as sp
import pandas as pd

corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey"
]


df = pd.read_csv(r'C:\Users\dmandava\Desktop\mtsamples.csv')
dft = df['transcription']
dfd = df['description']
dfs = df['sample_name']

data = []
for d in dft:
    data.append(str(d))

data = util.document_parser(data)
docs = util.preprocess(data)

#get filtered documents
#docs = util.document_parser(corpus)

words = set()
for sentence in docs:
    for token in sentence.split():
        words.add(token)

words = sorted(words)
print(words)
print(len(words))

#word_doc_freq = util.word_frequency(docs, words)

#give an index to the words
word_indexes = util.word_indexing(words)


def word_count(word, doc):
    c= 0
    for token in doc.split():
        if word == token:
            c+=1
    return c

#print(word_count('rohan','hi rohan and ravi'))
m = []

#for i,w in enumerate(words):
#    b=[]
 #   for j,d in enumerate(docs):
        #print("word:",i," document: ",j, word_count(w,d))
  #      b.append(word_count(w, d))
  #  m.append(b)
#m = np.array(m)
print("*****************")
#tfidf
vect = TfidfVectorizer()
X = vect.fit_transform(docs)
#vec = util.tfidf(docs, words, word_indexes, word_doc_freq)
#vec = np.asarray(vec)
#vec = vec.T

query = "trachea with ridged fiberoptic scopes"
#q_vec = np.zeros(len(word_indexes))
#for word in query.lower().split():
#    if word in words and q_vec[word_indexes[word]]==0:
#        q_vec[word_indexes[word]] = 1
#        #print("word: ", word, "word_index: ",word_indexes[word])
q_vec = vect.transform([query])
X = X.T
print(X.todense())
#u,s,vt = np.linalg.svd(X.todense())
u,s,vt = randomized_svd(X.todense(), n_components=300)
s = np.diag(s)
sinv = np.linalg.inv(s)

print("----------------")
#print("m: ", m.shape)
print("vec :", X.shape)
print("query shape: ",q_vec.shape)
print("u shape: ",u.shape)
print("s shape: ",s.shape)
print("v shape",vt.shape)
print("--------------------")


u = u
s = s


k = u@sinv
#q_vec = np.expand_dims(q_vec, axis = 1)

qt = q_vec@k

res = []
for i in range(4999):
    res.append(util.cosine_similarity(qt, vt[:, i]))

p = []
for row in res:
    p.append(row[0])
p = np.array(p)

d = np.array(d)
idx = np.arange(4999)[p>=0.3]

d = idx[np.argsort(-p[idx])]
print(d)

for i in d:
    print(dfs[i],"  ",dfd[i],end="\n")
