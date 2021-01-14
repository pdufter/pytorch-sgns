
import torch
import pickle
import os

vectors = torch.load("pts/sgns.pt")
with open("data/idx2word.dat", "rb") as fin:
	vocab = pickle.load(fin)

os.makedirs("export")
with open("export/ovectors.txt", "w") as fp:
	for row in vectors['embedding.ovectors.weight'].cpu().numpy():
		fp.write("\t".join([str(x) for x in row]) + "\n")

with open("export/ivectors.txt", "w") as fp:
	for row in vectors['embedding.ivectors.weight'].cpu().numpy():
		fp.write("\t".join([str(x) for x in row]) + "\n")

with open("export/words.txt", "w") as fp:
	for token in vocab:
		fp.write("{}\n".format(token))