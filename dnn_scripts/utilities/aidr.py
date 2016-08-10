from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter


def load_and_numberize_data(path="../data/", nb_words=None, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3, init_type="random", embfile=None, dev_train_merge=0, map_labels_to_five_class=0):

    """ numberize the train, dev and test files """

    # read the vocab from the entire corpus (train + test + dev)
    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test  = []
    y_test = []

    sentences_dev   = []
    y_dev  = []

    for filename in glob.glob(os.path.join(path, '*.csv')):
#        print "Reading vocabulary from" + filename
        reader  = csv.reader(open(filename, 'rb'))
	print (filename)
        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue
            if re.search("train.csv", filename.lower()):    
                sentences_train.append(row[1])
                y_train.append(row[2])    

            elif re.search("test.csv", filename.lower()):    
                sentences_test.append(row[1])    
                y_test.append(row[2])    

            elif re.search("dev.csv", filename.lower()):    
                sentences_dev.append(row[1])    
                y_dev.append(row[2])    

            for wrd in row[1].split():
                vocab[wrd] += 1

#    print (sentences_train)
    #print (y_test)
    #print (sentences_dev)

    print "Nb of tweets: train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev))
    print "Total vocabulary size: " + str (len(vocab))

    if nb_words is None: # now take a fraction
        nb_words = len(vocab) 
    else:
        pr_perc  = nb_words
        nb_words = int ( len(vocab) * (nb_words / 100.0) )    

#    if nb_words is None or nb_words > len(vocab):
#        nb_words = len(vocab) 

    vocab = dict (vocab.most_common(nb_words))

    print "Pruned vocabulary size: " + str (pr_perc) + "% =" + str (len(vocab))

    #Create vocab dictionary that maps word to ID
    vocab_list = vocab.keys()
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_train = numberize_sentences(sentences_train, vocab_idmap, oov_char=oov_char)
    X_test  = numberize_sentences(sentences_test,  vocab_idmap, oov_char=oov_char)
    X_dev   = numberize_sentences(sentences_dev,   vocab_idmap, oov_char=oov_char)


    #Create label dictionary that map label to ID
    merge_labels = None
    

    if map_labels_to_five_class:                

        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    #SUGGESTION
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        y_dev   = remap_labels(y_dev,   merge_labels=merge_labels)

    label_list = sorted( list (set(y_train)))

    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id  

    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)


    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data

    print("Random seed", str(seed))
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)


    X_train, y_train = adjust_index(X_train, y_train, start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_dev,   y_dev   = adjust_index(X_dev,   y_dev,   start_char=start_char, index_from=index_from, maxlen=maxlen)

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)
#        y_train=np.concatenate ((y_train, y_dev)) # need if y_train is numpy array
 
    # load the embeddeings
    if init_type.lower() != "random" and embfile:
        E = load_emb(embfile, vocab_idmap, index_from)
    else:
        E = None


    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), nb_words + index_from, E, label_map


def remap_labels(y, merge_labels=None):

    if not merge_labels:
        return y

    y_modified = []
    for alabel in y:
        if merge_labels.has_key(alabel):
            y_modified.append(merge_labels[alabel])
        else:
            y_modified.append(alabel)

    return y_modified        


def load_emb(embfile, vocab_idmap, index_from=3, start_char=1, oov_char=2, padding_char=0, vec_size=300):
    """ load the word embeddings """

    print "Loading pre-trained word2vec embeddings......"

    if embfile.endswith(".gz"):
        f = gzip.open(embfile, 'rb')
    else:
        f = open(embfile, 'rb')

    vec_size_got = int ( f.readline().strip().split()[1]) # read the header to get vec dim

    if vec_size_got != vec_size:
        print " vector size provided and found in the file don't match!!!"
        raw_input(' ')
        exit(1)

    # load Embedding matrix
    row_nb = index_from+len(vocab_idmap)    
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (row_nb, vec_size) )

    wrd_found = {}


    #fo = open("reduced_crisis_emb.txt", "rw+")
    #fo.write("300000 300\n")	
    #test = 0
    for line in f: # read from the emb file
        all_ent   = line.split()
	#print(all_ent)
        word, vec = all_ent[0].lower(), map (float, all_ent[1:])

        if vocab_idmap.has_key(word):
            wrd_found[word] = 1 
            wid    = vocab_idmap[word] + index_from
            E[wid] = np.array(vec)
	    #test = test + 1
	    #write to tmp file
	    #fo.write(line)
    #print(test)
    #fo.close()
    f.close()
    print " Number of words found in emb matrix: " + str (len (wrd_found)) + " of " + str (len(vocab_idmap))

    return E        




def adjust_index(X, labels, start_char=1, index_from=3, maxlen=None):

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if start_char is not None: # add start of sentence char
        X = [[start_char] + [w + index_from for w in x] for x in X] # shift the ids to index_from; id 3 will be shifted to 3+index_from
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)

        X      = new_X
        labels = new_labels

    return (X, labels)     


def numberize_sentences(sentences, vocab_idmap, oov_char=2):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        for wrd in sent.split():
            wrd_id = vocab_idmap[wrd] if vocab_idmap.has_key(wrd) else oov_char 
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id    


def numberize_labels(all_str_label, label_id_map):

    label_cnt = {}
    labels    = []
    
    for a_label in all_str_label:
        labels.append(label_id_map[a_label])

        if label_cnt.has_key(a_label):
            label_cnt[a_label] += 1
        else:
            label_cnt[a_label] = 1    

    return (labels, label_cnt)        



def get_label(str_label):
    if  str_label == "informative":
        return 1
    elif str_label == "not informative":
        return 0
    else:
        print "Error!!! unknown label " + str_label
        exit(1)        


def load_data(path="imdb.pkl", nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):


    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    X, labels = cPickle.load(f)
    f.close()


    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels

    if not nb_words:
        nb_words = max([max(x) for x in X])


    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX


    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)
