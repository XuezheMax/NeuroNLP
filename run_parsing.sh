THEANO_FLAGS='floatX=float32' python parsing.py --num_epochs 200 --batch_size 100 --num_units 100 --num_filters 30 \
 --learning_rate 0.01 --decay_rate 0.1 --schedule 50 100 150 --grad_clipping 5 --regular none --dropout 0.5 --delta 0.0 --punctuation ", . \`\` : ''" \
 --train "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.train.conll" \
 --dev "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.dev.conll" \
 --test "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.test.conll" 
