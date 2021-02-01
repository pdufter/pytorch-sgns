
python create_fake.py --input data/corpus.txt --output data/eng_kingjames_fake.txt


python preprocess.py --window 2 --max_vocab 4000


python train.py --weights --cuda --epoch 1 --n_negs 5 --e_dim 2 --hidden 300 --multilingual --tie_weights
python train.py --weights --cuda --epoch 100 --n_negs 5 --e_dim 100
# okish python train.py --weights --cuda --epoch 10 --n_negs 10 --e_dim 20 --hidden 300 --multilingual --sample_within --lr 0.001


python evaluate.py --vectors pts/sgns.pt --words data/idx2word.dat

python export.py


# train fasttext
nice -n 19 /mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext skipgram \
-input data/corpus.txt \
-output embeddings/fasttext.txt \
-dim 300 \
-thread 20