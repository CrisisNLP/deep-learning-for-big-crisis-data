'''Train LSTM RNNs on the AIDR tweet classification task.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_rnns_aidr.py

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

# keras related
from keras.models import Sequential
from keras.layers.core    import Dense, Dropout, Activation, Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from utilities import aidr
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json

#other utilities 
import optparse
import logging
import sys
import csv
import os
csv.field_size_limit(sys.maxsize)



from sklearn import metrics





def build_cnn(maxlen, max_features, emb_size=128, emb_matrix=None, nb_filter=250, filter_length=3,
			 pool_length=2,	nb_classes = 2,  hidden_size=128, dropout_ratio=0.5, tune_emb=True):

	''' build cnn model '''

	print('Building model:', 'convolutional neural network (cnn)')

	#create the emb layer	
	if emb_matrix is not None:
		max_features, emb_size = emb_matrix.shape
		emb_layer = Embedding(max_features, emb_size, weights=[emb_matrix], input_length=maxlen, trainable=tune_emb)

	else:
		emb_layer = Embedding(max_features, emb_size, input_length=maxlen, trainable=tune_emb)


	model = Sequential()
	model.add(emb_layer)	
	model.add(Dropout(dropout_ratio))

	# we add a Convolution1D, which will learn nb_filter (word group) filters of size filter_length:
	model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, 
                        	border_mode='valid', activation='relu', subsample_length=1))

	# we use standard max pooling (halving the output of the previous layer):
	model.add(MaxPooling1D(pool_length=pool_length))
	model.add(Dropout(dropout_ratio))

	# We flatten the output of the conv layer, so that we can add a vanilla dense layer:
	model.add(Flatten())

	# We add a vanilla hidden layer:
	model.add(Dense(hidden_size))
	model.add(Activation('relu'))
	model.add(Dropout(dropout_ratio))

	# We project onto a single unit output layer, and squash it with a sigmoid:

	if nb_classes == 2:
		print('Doing binary classification...')
		model.add(Dense(1))
		model.add(Activation('sigmoid'))

	elif nb_classes > 2:
		print('Doing classification with class #', nb_classes)
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))
	else:
		print("Wrong argument nb_classes: ", nb_classes)
		exit(1)

	return model


if __name__ == '__main__':


	logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

	# parse user input
	parser = optparse.OptionParser("%prog [options]")

	#file related options
	parser.add_option("-g", "--log-file",          dest="log_file", help="log file [default: %default]")
	parser.add_option("-d", "--data-dir",          dest="data_dir", help="directory containing train, test and dev file [default: %default]")
	parser.add_option("-D", "--data-spec",         dest="data_spec", help="specification for training data (in, out, in_out) [default: %default]")
	parser.add_option("-m", "--model-dir",         dest="model_dir", help="directory to save the best models [default: %default]")

#	parser.add_option("-r", "--train-file",        dest="featFile_train")
#	parser.add_option("-s", "--test-file",         dest="featFile_test")
#	parser.add_option("-v", "--validation-file",   dest="featFile_dev")

	# network related
	parser.add_option("-t", "--max-tweet-length",  dest="maxlen",       type="int", help="maximul tweet length (for fixed size input) [default: %default]") # input size

	parser.add_option("-F", "--nb_filter",         dest="nb_filter",     type="int",   help="nb of filter to be applied in convolution over words [default: %default]") # uni, bi-directional
	parser.add_option("-r", "--filter_length",     dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]") # lstm, gru, simpleRNN
	parser.add_option("-p", "--pool_length",       dest="pool_length",   type="int",   help="length for max pooling [default: %default]") # lstm, gru, simpleRNN
	parser.add_option("-v", "--vocabulary-size",   dest="max_features",  type="float",   help="vocabulary size in percentage [default: %default]") # emb matrix row size
	parser.add_option("-e", "--emb-size",          dest="emb_size",      type="int",   help="dimension of embedding [default: %default]") # emb matrix col size
	parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int",   help="hidden layer size [default: %default]") # size of the hidden layer
	parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

	parser.add_option("-i", "--init-type",         dest="init_type",     help="random or pretrained [default: %default]") 
	parser.add_option("-f", "--emb-file",          dest="emb_file",      help="file containing the word vectors [default: %default]") 
	parser.add_option("-P", "--tune-emb",          dest="tune_emb",      action="store_false", help="DON't tune word embeddings [default: %default]") 

	#learning related
	parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
	parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
	parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
	parser.add_option("-n", "--epochs",             dest="epochs", type="int", help="nb of epochs [default: %default]")


	parser.set_defaults(
#    	data_dir        = "../data/"
    	data_dir        = "../data/earthquakes/in/"
    	,data_spec       = "in"
	    ,log_file       = "log"
    	,model_dir      = "./saved_models/"

#    	,featFile_train = "../data/good_vs_bad/CQA-QL-train.xml.multi.csv.feat"
#    	,featFile_test  = "../data/good_vs_bad/CQA-QL-test.xml.multi.csv.feat"
#    	,featFile_dev   = "../data/good_vs_bad/CQA-QL-devel.xml.multi.csv.feat"

	   	,learn_alg      = "adadelta" # sgd, adagrad, rmsprop, adadelta, adam (default)
	   	,loss           = "binary_crossentropy" # hinge, squared_hinge, binary_crossentropy (default)
	    ,minibatch_size = 32
    	,dropout_ratio  = 0.0

    	,maxlen         = 100
    	,epochs         = 25
    	,max_features   = 80
    	,emb_size       = 128
    	,hidden_size    = 128
    	,nb_filter      = 250
    	,filter_length  = 3 
    	,pool_length    = 2 
    	,init_type      = 'random' 
    	,emb_file       = "../data/unlabeled_corpus.vec"
    	,tune_emb       = True
	)

	options,args = parser.parse_args(sys.argv)


	print('Loading data...')
	(X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id = aidr.load_and_numberize_data(path=options.data_dir,
																			nb_words=options.max_features, init_type=options.init_type,
																			embfile=options.emb_file, dev_train_merge=0, map_labels_to_five_class=0)

#	print("Padding sequences....")
	X_train = sequence.pad_sequences(X_train, maxlen=options.maxlen)
	X_test  = sequence.pad_sequences(X_test,  maxlen=options.maxlen)
	X_dev   = sequence.pad_sequences(X_dev,   maxlen=options.maxlen)


	#build model...
	nb_classes = np.max(y_train) + 1

	print('............................')
	print(len(X_train), 'train tweets')
	print(len(X_test),  'test  tweets')
	print(len(X_dev),   'dev   tweets')
	print(max_features - 3, 'vocabulary size')
	print(nb_classes, 'different classes')
	print('............................')


	if nb_classes == 2: # binary
		loss       = options.loss
		class_mode = "binary"
		optimizer  = options.learn_alg

	elif nb_classes > 2: # multi-class
		loss       = 'categorical_crossentropy'
		class_mode = 'categorical'
		optimizer  = options.learn_alg
		print("** optimizer: " + options.learn_alg)	
		# convert class vectors to binary class matrices [ 1 of K encoding]
		y_train_mod = np_utils.to_categorical(y_train, nb_classes)
		y_test_mod  = np_utils.to_categorical(y_test,  nb_classes)
		y_dev_mod   = np_utils.to_categorical(y_dev,   nb_classes)


	model = build_cnn(options.maxlen, max_features, emb_matrix=E, emb_size=options.emb_size, nb_filter=options.nb_filter,
						filter_length=options.filter_length, pool_length=options.pool_length, nb_classes = nb_classes,
						hidden_size=options.hidden_size, dropout_ratio=options.dropout_ratio, tune_emb=options.tune_emb)

	model.compile(optimizer=optimizer, loss=loss,  class_mode=class_mode)

	model_name = options.model_dir + "cnn" + "-" + optimizer + "-" + str(options.nb_filter) + "-" + str(options.filter_length) + \
        "-" + str(options.pool_length) + "-" + str (options.tune_emb) +\
	"-" + loss + "-" + str (options.minibatch_size) + "-" + str(options.dropout_ratio) + "-init-" + str (options.init_type) + "-" +\
	str (options.max_features) + "-" + str (options.emb_size) + "-" + str (options.hidden_size) + ".model.cl." + str(nb_classes) + ".dom." + str(options.data_spec) 

	earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
	checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)


	if nb_classes == 2: # binary
		print ('Training and validating ....')
		model.fit(X_train, y_train, batch_size=options.minibatch_size, nb_epoch=options.epochs,
				validation_data=(X_dev, y_dev), show_accuracy=True, verbose=2, callbacks=[earlystopper, checkpointer])

		print("Test model ...")
		print ("Loading ...", model_name)
		model.load_weights(model_name)

#		score, acc = model.evaluate(X_test, y_test, batch_size=options.minibatch_size, show_accuracy=True)
#		print('Test accuracy:', acc)

		y_prob = model.predict_proba(X_test)
		##added by kamla
		#print("Predictions")
		#for e in y_prob: print(e)
		###
		roc = metrics.roc_auc_score(y_test, y_prob)
		print("ROC Prediction (binary classification):", roc)


	elif nb_classes > 2: # multi-class
		print ('Training and validating ....')
                #check if there is pre-trained model
                #if os.path.exists(model_name) == False:
                #else:
                    #print("Loading pre-trained model...")
                    #model = model_from_json(open(model_name + ".json").read())
                    #model.load_weights(model_name)
                    #model.compile(optimizer=optimizer, loss=loss, class_mode=class_mode) 
		model.fit(X_train, y_train_mod, batch_size=options.minibatch_size, nb_epoch=options.epochs,
				validation_data=(X_dev, y_dev_mod), show_accuracy=True, verbose=2, callbacks=[earlystopper, checkpointer])

		print ("Loading ...", model_name)
		model.load_weights(model_name)       
                print("Test model ...")

	y_pred = model.predict_classes(X_test)
	y_test = np.array(y_test)

	acc2 = metrics.accuracy_score(y_test, y_pred)
	print("Raw Accuracy:", acc2)

	#get label ids in sorted
	class_labels = sorted(label_id, key=label_id.get)
	#print (class_labels)

	print (metrics.classification_report(y_test, y_pred, target_names=class_labels, digits=4) )

	print ("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred, labels=range(0, len(class_labels))))

	if nb_classes == 2:
		_p, _r, _f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
		print (" pre: " + str (_p) + " rec: " + str (_r) + " f-score: " + str (_f))
           
	else:
		mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='micro')
		mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')
		print (" micro pre: " + str (mic_p) + " rec: " + str (mic_r) + " f-score: " + str (mic_f))
		print (" macro pre: " + str (mac_p) + " rec: " + str (mac_r) + " f-score: " + str (mac_f))

        # save the architecture finally in json format
        json_string = model.to_json()
        open(model_name + ".json", 'w').write(json_string)




