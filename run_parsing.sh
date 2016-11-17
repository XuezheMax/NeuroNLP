THEANO_FLAGS='floatX=float32' python parsing.py --num_epochs 200 --batch_size 200 --num_units 300 --num_filters 50 --normalize_digits --pos \
 --opt adam --learning_rate 0.002 --decay_rate 0.75 --schedule 2000 --grad_clipping 5.0 --regular none --dropout 0.25 \
 --punctuation ", . \`\` : ''" \
 --train "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.train.conll" \
 --dev "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.dev.conll" \
 --test "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.test.conll" \
 --tmp "tmp"
