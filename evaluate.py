import argparse
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_distances


def get_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cosine_distances(X, Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default=None, type=str, required=True, help="")
    parser.add_argument("--words", default=None, type=str, required=True, help="")
    args = parser.parse_args()
    import ipdb;ipdb.set_trace()
    vectors = torch.load(args.vectors)
    with open("data/idx2word.dat", "rb") as fin:
        vocab = np.array(pickle.load(fin))
    real = sorted([x for x in vocab if not x.startswith("::")])
    fake = sorted([x for x in vocab if x.startswith("::")])
    real_indices = np.array([vocab.index(x) for x in real])
    fake_indices = np.array([vocab.index(x) for x in fake])
    vectors_real = vectors[real_indices]
    vectors_fake = vectors[fake_indices]

    dist = get_distances(vectors_real, vectors_fake)
    if dist.shape[0] != dist.shape[1]:
        print("Number of words is different?")
    # get different p@k
    nns = np.argsort(dist, axis=1)[:, :10]
    # import ipdb;ipdb.set_trace()
    gt = np.arange(dist.shape[0]).reshape(-1, 1)
    p = {}
    for considern in [1, 5, 10]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        p[considern] = hits1 / dist.shape[0]
    nns = np.argsort(dist, axis=0)[:10, :].transpose()
    gt = np.arange(dist.shape[0]).reshape(-1, 1)
    pinv = {}
    for considern in [1, 5, 10]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        pinv[considern] = hits1 / dist.shape[0]
    return p, pinv



if __name__ == '__main__':
    main()
