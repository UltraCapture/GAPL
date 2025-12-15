import torch
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import numpy as np
import time

from new_dataset import RealFakeDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_class_centroids(embeddings, labels, normalize=True):
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
    centroids = {}
    for c in np.unique(labels):
        Xc = embeddings[labels == c]
        centroids[c] = Xc.mean(axis=0)
        centroids[c] /= np.linalg.norm(centroids[c]) + 1e-12
    return centroids

def get_class_prototypes_kmeans(embeddings, labels, m_per_class=3, normalize=True, random_state=0):
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
    prototypes = []
    for c in np.unique(labels):
        if c == 0:
            k = 8
        else:
            k = min(m_per_class, len(Xc))
        Xc = embeddings[labels == c]
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(Xc)
        prototypes.append(kmeans.cluster_centers_)
    return prototypes

def initialize_prototypes_with_pca(features, num_prototypes: int):
    if isinstance(features, torch.Tensor):
        if features.is_cuda:
            features = features.cpu()
        features_np = features.numpy()
    elif isinstance(features, np.ndarray):
        features_np = features
    else:
        raise TypeError(f"Not valid type {type(features)}")

    num_samples, feature_dim = features_np.shape
    print(f"Starting PCA with {num_samples} samples，dim {feature_dim}...")
    print(f"target number: {num_prototypes}")

    start_time = time.time()
    pca = PCA(n_components=num_prototypes)
    pca.fit(features_np)
    end_time = time.time()
    print(f"PCA done，Time: {end_time - start_time:.4f} S")

    prototypes_np = pca.components_
    prototypes_tensor = torch.tensor(prototypes_np, dtype=torch.float32)

    return prototypes_tensor

def getOrthVec(P, dim):
    new_p = np.random.randn(dim)
    for u in P:
        new_p = new_p - np.dot(u, new_p) * u
        new_p = new_p / np.linalg.norm(new_p)
    return new_p

if __name__ == "__main__":
    ckpt_path = "./trained_model/20251009-055952CLIP_freeze_ProGAN/checkpoints/checkpoint_9.pt"
    num_prototype = 64

    model = models.ClipModel(freeze_backbone=True, num_classes=1).cuda()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.aproj.load_state_dict(ckpt['aproj'])

    ds = RealFakeDataset()
    ds.summary()
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)

    embeddings = []
    labels = []

    for img, label in dl:
        with torch.no_grad():
            img = img.cuda()
            feat = model.feature_extractor(img)['pooler_output']
            feat = model.aproj(feat)
            feat = F.normalize(feat, p=2)
            embeddings.append(feat.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    r_mask = (labels == 0)
    f_mask = (labels == 1)

    r_pro = initialize_prototypes_with_pca(embeddings[r_mask], num_prototypes=num_prototype // 2)
    f_pro = initialize_prototypes_with_pca(embeddings[f_mask], num_prototypes=num_prototype // 2)

    proVec = torch.tensor(np.concatenate([r_pro, f_pro], axis=0))
    proVec = F.normalize(proVec, dim=1)

    torch.set_printoptions(precision=2, sci_mode=False)

    print(proVec.shape)
    print(proVec @ proVec.T)
    torch.save(proVec, f'./prototype/pca_3type_num{num_prototype}.pt')


