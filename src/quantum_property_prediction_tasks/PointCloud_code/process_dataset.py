import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def load_data(dataset, partition, npoint):
    point_data = pickle.load(open(f"./{dataset}/processed/mol_EDthresh0_data.pkl", 'rb'))
    label_data = pd.read_csv(f"./{dataset}/raw/{dataset}.csv")

    labels = label_data[label_data["scaffold_split"] == partition]["label"].tolist()
    try:
        labels = [list(map(float, x.split(" "))) for x in labels]
        labels = np.array(labels, dtype=np.float32)
    except:
        # classfication task
        labels = np.array(labels, dtype=np.int64)

    idxs = label_data[label_data["scaffold_split"] == partition]["index"].tolist()

    coords, densitys = [], []

    for idx in tqdm(idxs):
        coord = point_data[f"{idx}"]["electronic_density"]["coords"]
        d = point_data[f"{idx}"]["electronic_density"]["density"]
        process_point = np.column_stack((coord, d))

        point = farthest_point_sample(process_point, npoint)

        coords.append(point)

    return coords, labels


def main(args):
    dataset = args.dataset
    data_dict = {}
    npoint = args.npoint

    for split in ["train", "test", "valid"]:
        coords, labels = load_data(dataset, split, npoint)
        data_dict[split] = {
            "coords": coords,
            "labels": labels
        }

    with open(f"./{dataset}/processed/{dataset}_{npoint}.pkl", "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == '__main__':
    parser = ArgumentParser(description='Pre-process of the dataset about npoint')

    # basic
    parser.add_argument('--dataset', type=str, default="ed_homo_lumo_5w", help='dataset')
    parser.add_argument('--npoint', type=int, default=1024, help='npoint')

    args = parser.parse_args()
    main(args)

