import torch
import logging
import numpy as np
from tqdm import tqdm
import faiss
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import time

def test(args, eval_ds, model, pca=None):
    model = model.eval()

    query_num = eval_ds.queries_num
    gallery_num = eval_ds.database_num
    if(args.n_gpus>1):
        all_features = np.empty((query_num + gallery_num, model.module.aggregation_dim), dtype=np.float32)
    else:
        all_features = np.empty((query_num + gallery_num, model.aggregation_dim), dtype=np.float32)

    with torch.no_grad():
        logging.debug("Extracting gallery features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=4,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        for images, indices, _ in tqdm(database_dataloader, ncols=100):
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            _, frames_features = model(images.to(args.device))#改
            if(args.n_gpus>1):
                aggregated_features = model.module.aggregate(frames_features)
            else:
                aggregated_features = model.aggregate(frames_features)
            if isinstance(aggregated_features, list):
                aggregated_features = aggregated_features[0].cpu().numpy()
            else:
                aggregated_features = aggregated_features.cpu().numpy()
            if pca:
                aggregated_features = pca.transform(aggregated_features)
            all_features[indices.numpy(), :] = aggregated_features

        logging.debug("Extracting queries features for evaluation/testing")
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=4,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        for images, _, indices in tqdm(queries_dataloader, ncols=100):
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            _, frames_features = model(images.to(args.device))#改
            if(args.n_gpus>1):
                aggregated_features = model.module.aggregate(frames_features)
            else:
                aggregated_features = model.aggregate(frames_features)
            if isinstance(aggregated_features, list):
                aggregated_features = aggregated_features[0].cpu().numpy()
            else:
                aggregated_features = aggregated_features.cpu().numpy()
            if pca:
                aggregated_features = pca.transform(aggregated_features)
            all_features[indices.numpy(), :] = aggregated_features

    torch.cuda.empty_cache()
    queries_features = all_features[eval_ds.database_num:]
    gallery_features = all_features[:eval_ds.database_num]
    if(args.n_gpus>1):
        faiss_index = faiss.IndexFlatL2(model.module.aggregation_dim)
    else:
        faiss_index = faiss.IndexFlatL2(model.aggregation_dim)
    faiss_index.add(gallery_features)

    logging.debug("Calculating recalls")
    
    _, predictions = faiss_index.search(queries_features, 10)
    positives_per_query = eval_ds.pIdx
    recall_values = [1, 5, 10]
    recalls = np.zeros(len(recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / len(eval_ds.qIdx) * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
    return recalls, recalls_str
