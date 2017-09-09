THEANO_FLAGS='floatX=float32' python bi-lstm-cnn.py --num_epochs 200 --batch_size 16 --num_units 200 --num_filters 30 \
 --learning_rate 0.01 --decay_rate 0.05 --schedule 10 20 40 80 150 --grad_clipping 0 --regular none --dropout recurrent \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
