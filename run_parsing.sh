THEANO_FLAGS='floatX=float32' python parsing.py --num_epochs 200 --batch_size 200 --num_units 300 --num_filters 50 --normalize_digits --pos \
 --opt adam --learning_rate 0.002 --decay_rate 0.75 --schedule 2000 --grad_clipping 5.0 --regular none --dropout 0.25 \
 --punctuation "," "." "\`\`" ":" "''" --embedding sskip --embedding_dict "data/sskip/sskip.chn.50.gz" \
 --train "data/CTB-SD/ctb.train.conll" \
 --dev "data/CTB-SD/ctb.dev.conll" \
 --test "data/CTB-SD/ctb.test.conll" \
 --tmp "tmp"
