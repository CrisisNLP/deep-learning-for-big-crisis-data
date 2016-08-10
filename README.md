# Deep Learning for Big Crisis Data
This repository will host Python implementation of a number of deep neural networks classifiers
for the classification of crisis-related data on Twitter.

1. Requirementes:
        
        python 2.7
        numpy, scikit-learn
        keras, tensorflow or theano backend

2. Dataset and Pre-process
	A sample of tweet data (data/sample.csv) is a .csv format with three columns  
	
        First, we need to pre-process tweets data: remove urls, special characters, lowercasing…
    
    	- python data_helpers/preprocess.py data/sample.csv
        
        Split pre-processed data (data/sample_prccd.csv) into train, test and dev part.
	
        - python data_helpers/split_data.py data/sample_prccd.csv
	  
3. Training a neural net model

	To train a classifier we create a folder containing links to train, test and dev part (data/nn_data)

	Folder embeddings/ includes word vector file, we provide our pre-trained crisis word vectors, we also can use Google word embedding here

	Folder dnn_scrips/ contains all neural nets models: CNN, RNN_LSTM, MLP…

	- bash run_cnn.sh to train a model with different parameters.

	See the results and training process in .log file

        Note that: if you do not have a GPU, you can run on CPU by change the theano flag to THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32
