THEANO_FLAGS='floatX=float32' python sent_classify.py --architec sgru --num_epochs 100 --batch_size 16 --num_units 64 \
 --learning_rate 0.001 --decay_rate 0.1 --schedule 30 70 --grad_clipping 0 
