#=================
#==> Libraries <==
#=================
import sys, os
import csv
import numpy as np
from sklearn.cross_validation import StratifiedKFold


#============
#==> Main <==
#============

#-----------------
#--> Load data <--
#-----------------
in_file=sys.argv[1]

with open(in_file, 'rU') as fin:
	i=np.array([])
	X=np.array([])
	y=np.array([])
        rows = csv.reader(fin)
	header = next(rows)
	for col in rows:
		i = np.append(i,col[0]) #tweet_id
		X = np.append(X,col[1]) #tweet
		y = np.append(y,col[2]) #label

#------------------------------------------
#--> split data into train+dev and test <--
#------------------------------------------
skf = StratifiedKFold(y, n_folds=5, shuffle=True) # n_folds = sum of train to dev to test ratio

for train_dev_index, test_index in skf:
	train_dev = zip(i[train_dev_index], X[train_dev_index], y[train_dev_index])
	test = zip(i[test_index], X[test_index], y[test_index])

#-----------------------------------------
#--> intermidate files: train+dev file <--
#-----------------------------------------
filename = os.path.splitext(in_file)[0]
filename = os.path.basename(filename)

with open("%s_train_dev.csv" %filename, "wb") as f:
	writer = csv.writer(f)
	writer.writerow(header)
	writer.writerows(train_dev)

#---------------------------
#--> load train_dev data <--
#---------------------------
with open("%s_train_dev.csv" %filename) as fin:
        ii=np.array([])
        XX=np.array([])
        yy=np.array([])
        rows = csv.reader(fin)
        header = next(rows)
        for col in rows:
                ii = np.append(ii,col[0]) #tweet_id
                XX = np.append(XX,col[1]) #tweet
                yy = np.append(yy,col[2]) #label

#------------------------------------------
#--> split train+dev into train and dev <--
#------------------------------------------

skf = StratifiedKFold(yy, n_folds=8, shuffle=True) # n_folds = sum of train to dev ratio

for train_index, dev_index in skf:
        train = zip(ii[train_index], XX[train_index], yy[train_index])
        dev = zip(ii[dev_index], XX[dev_index], yy[dev_index])

#--------------------
#--> save outputs <--
#--------------------

with open("%s_train.csv" %filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train)

with open("%s_dev.csv" %filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dev)

with open("%s_test.csv" %filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test)

#---------------------------------
#--> deleter intermidate files <--
#---------------------------------
