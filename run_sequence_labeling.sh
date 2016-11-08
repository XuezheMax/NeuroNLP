THEANO_FLAGS='floatX=float32' python sequence_labeling.py --num_epochs 200 --batch_size 10 --num_units 200 --num_filters 30 \
 --learning_rate 0.01 --decay_rate 0.1 --schedule 50 100 150 --grad_clipping 5 --regular none --dropout recurrent --delta 1.0 \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
