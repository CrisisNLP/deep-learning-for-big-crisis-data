CNN_SCR="./dnn_scripts/cnn_crisis.py"
MODEL_DIR="saved_models/"

data=./data/nn_data/

#log=./mix-in-domain-embGG.log
log=./log.cnn


mkdir -p $MODEL_DIR

###<- Set general DNN settings ->
dr_ratios=(0.2) #dropout_ratio
mb_sizes=(128) #minibatch-size

### <- set CNN settings ->
nb_filters=(150) #no of feature map
filt_lengths=(2)
pool_lengths=(3) 

vocab_sizes=(90) # how many words in percentage for vocabulary

### <- embedding file ->
init_type="pretrained"
emb_file="./embeddings/crisis_embeddings.text"

for ratio in ${dr_ratios[@]}; do
	for mb in ${mb_sizes[@]}; do
		for nb_filter in ${nb_filters[@]}; do
			for filt_len in ${filt_lengths[@]}; do
				for pool_len in ${pool_lengths[@]}; do
					for vocab in ${vocab_sizes[@]}; do
							echo "INFORMATION: dropout_ratio=$ratio minibatch-size=$mb filter-nb=$nb_filter filt_len=$filt_len pool_len=$pool_len vocab=$vocab" >> $log;
							echo "----------------------------------------------------------------------" >> $log;

							THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python $CNN_SCR \
							--data-dir=$data --model-dir=$MODEL_DIR -i $init_type -f $emb_file\
							--vocabulary-size=$vocab --dropout_ratio=$ratio --minibatch-size=$mb\
							--nb_filter=$nb_filter --filter_length=$filt_len --pool_length=$pool_len\
							--vocabulary-size=$vocab  >>$log
							wait

							echo "----------------------------------------------------------------------" >> $log;

					done
				done
			done
		done 
	done	
done 
