python preprocess.py
python train.py --weights --cuda --epoch 100 --n_negs 5 --e_dim 100

python create_fake.py --input data/corpus.txt --output data/eng_kingjames_fake.txt