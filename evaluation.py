import logging
from datetime import datetime
import torch

from jist.datasets import BaseDataset,PCADataset_sequence
from jist import utils, evals
from jist.models import DJIST

import numpy as np#pca
from sklearn.decomposition import PCA#pca
import random#pca
from torch.utils.data.dataset import Subset#pca
from tqdm import tqdm#pca
import einops#pca
#from thop import profile#FLOPs, params

def compute_pca(args, model, transform, full_features_dim):
    model = model.eval()
    pca_ds = PCADataset_sequence(dataset_folder=args.seq_dataset_path, split='train',
                        base_transform=transform, seq_len=args.seq_length)
    logging.info(f'PCA dataset: {pca_ds}')
    num_images = min(len(pca_ds), 2 ** 14)
    if num_images < len(pca_ds):
        idxs = random.sample(range(0, len(pca_ds)), k=num_images)
    else:
        idxs = list(range(len(pca_ds)))
    subset_ds = Subset(pca_ds, idxs)
    dl = torch.utils.data.DataLoader(subset_ds, args.infer_batch_size)

    pca_features = np.empty([num_images, full_features_dim])
    with torch.no_grad():
        for i, sequences in enumerate(tqdm(dl, ncols=100, desc="Database sequence descriptors for PCA: ")):
            if len(sequences.shape) == args.seq_length:
                sequences = einops.rearrange(sequences, "b s c h w -> (b s) c h w")
            _, frames_features = model(sequences.to(args.device))
            aggregated_features = model.aggregate(frames_features)
            if isinstance(aggregated_features, list):
                aggregated_features = aggregated_features[0].cpu().numpy()
            else:
                aggregated_features = aggregated_features.cpu().numpy()
            pca_features[i * args.infer_batch_size : (i * args.infer_batch_size ) + len(aggregated_features)] = aggregated_features
    pca = PCA(args.pca_dim)
    logging.info(f'Fitting PCA from {full_features_dim} to {args.pca_dim}...')
    pca.fit(pca_features)
    return pca

def evaluation(args):
    start_time = datetime.now()
    args.output_folder = f"test/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    utils.setup_logging(args.output_folder, console="info")
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")

    ### Definition of the model
    model = DJIST(args, agg_type=args.aggregation_type)

    if args.resume_model != None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device)

    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_shape = (args.img_shape[0], args.img_shape[1])
    transform = utils.configure_transform(image_dim=img_shape, meta=meta)
    
    if args.pca_dim is None:
        pca = None
    else:
        full_features_dim = model.aggregation_dim
        model.aggregation_dim = args.pca_dim
        pca = compute_pca(args, model, transform, full_features_dim)

    eval_ds1 = BaseDataset(dataset_folder=args.seq_dataset_path, split=args.test_ds_city1,
                          base_transform=transform, seq_len=args.seq_length,
                          pos_thresh=args.val_posDistThr, reverse_frames=args.reverse)
    if (args.test_ds_city2!='none'):
        eval_ds2 = BaseDataset(dataset_folder=args.seq_dataset_path, split=args.test_ds_city2,
                          base_transform=transform, seq_len=args.seq_length,
                          pos_thresh=args.val_posDistThr, reverse_frames=args.reverse)
        logging.info(f"Test set: {eval_ds1} - Test set2: {eval_ds2} ")
    else : logging.info(f"Test set: {eval_ds1}")

    logging.info(f"Backbone output channels are {model.features_dim}, features descriptor dim is {model.fc_output_dim}, "
             f"sequence descriptor dim is {model.aggregation_dim}")

    _, recalls_str1 = evals.test(args, eval_ds1, model,pca)
    if (args.test_ds_city2!='none'):
        _, recalls_str2 = evals.test(args, eval_ds2, model,pca)
        logging.info(f"Recalls on test set: {recalls_str1} - test set2: {recalls_str2}") 
    else : logging.info(f"Recalls on test set: {recalls_str1}") 
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")


if __name__ == "__main__":
    args = utils.parse_arguments()
    evaluation(args)
