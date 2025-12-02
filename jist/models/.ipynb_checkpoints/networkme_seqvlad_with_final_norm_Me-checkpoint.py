
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from . import helper
import math
import numpy as np
import einops
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, SubsetRandomSampler
import faiss
from tqdm import tqdm
import logging

class Tokenmixen(nn.Module):
    def __init__(self, fc_in_channels=768, in_dim=16*16, mlp_ratio=1):
        super().__init__()
        self.norm = nn.LayerNorm(fc_in_channels)
        self.mix = nn.Sequential(
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
            #nn.Dropout(0.1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(self.norm(x.permute(0,2,1)).permute(0,2,1))

class SeqVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, seq_length, clusters_num=64, dim=128, normalize_input=True, transf_backbone=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.features_dim = dim * clusters_num
        self.transf_backbone = transf_backbone
        if transf_backbone:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))
        self.seq_length = seq_length

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.transf_backbone:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.transf_backbone:
            x = einops.rearrange(x, '(b s) d c -> b c (s d)', s=self.seq_length)
            N, D, _ = x.shape[:]
        else:
            x = einops.rearrange(x, '(b s) c h w -> b c (s h) w', s=self.seq_length)
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[D:D + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D + 1, :].unsqueeze(2)
            vlad[:, D:D + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_seqvlad_layer(self, args, cluster_ds, encoder, save_to_file=False):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_images = np.random.choice(cluster_ds.queries_num + cluster_ds.database_num, images_num, replace=False)
        subset_ds = Subset(cluster_ds, random_images)
        loader = DataLoader(dataset=subset_ds, num_workers=4,
                            batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        features_dim = self.features_dim // 64
        with torch.no_grad():
            encoder = encoder.eval()
            logging.debug("Extracting features to initialize SeqVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
            for iteration, (inputs, _, _) in enumerate(tqdm(loader, ncols=100)):
                inputs = inputs.to(args.device).view(-1, 3, args.img_shape[0], args.img_shape[1])
                inputs = inputs[::args.seq_length]  # take only first frame of each sequence
                _, outputs = encoder(inputs)
                #if isinstance(encoder, ViTModel):
                #    outputs = outputs.last_hidden_state[:, 1:, :]
                if self.transf_backbone:
                    # if using a transformer backbone, normalization is done token-wise
                    norm_outputs = F.normalize(outputs, p=2, dim=2)
                else:
                    norm_outputs = F.normalize(outputs, p=2, dim=1)

                image_descriptors = norm_outputs.view(norm_outputs.shape[0], features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(features_dim, 64, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"SeqVLAD centroids shape: {kmeans.centroids.shape}")
        # if save_to_file: # or args.save_centroids:
        #     self.save_centroids(kmeans.centroids, descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

class GeMPool(nn.Module):#这个gem pool的norm顺序跟crica不一样，不知道会不会有影响
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        return x


class Me(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, backbone_arch='dinov2_vitb14', pretrained=True, layer1=11, use_cls=False, norm_descs=True,out_indices=[8, 9, 10, 11],backbone_out_dim=3072,mix_in_dim=768,token_num=2,token_ratio=1):
        super().__init__()
        # get the backbone and the aggregator,先用默认的backbone参数
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1,  use_cls, norm_descs, out_indices)
        self.conv = nn.Conv2d(backbone_out_dim, mix_in_dim, (1, 1))#heatmap,维度待确定
        self.relu = nn.ReLU(inplace=False)
        if(token_num!=0):
            self.tokenmix = nn.Sequential(*[Tokenmixen(mix_in_dim, 16*16, token_ratio) for _ in range(token_num)])
        else:
            self.tokenmix = nn.Identity()
        #self.gemfc = GeMFc()

        self.gem = GeMPool()
        self.fc = nn.Linear(768, 512)


        #self.init_conv()

    #def init_conv(self):
    #    nn.init.constant_(self.conv.weight, 1.0 / 768)
    #    nn.init.zeros_(self.conv.bias)
    #    nn.init.constant_(self.conv_rev.weight, 1.0 / 768)
    #    nn.init.zeros_(self.conv_rev.bias)

    def forward(self, x):
        x = self.backbone(x)#B,C0,H,W    

        x = self.conv(x)#B,C,H,W   
        x = self.relu(x).flatten(2)
        x = self.tokenmix(x)
        
        #x = x.flatten(2)
        #x = self.relu(x)
        #heatmaps = normalization(x.sum(1))#B,H,W
        #print(heatmaps)
        #print(heatmaps.shape)
        #x_global = self.gem(x1, avg_pool2d = True) #try1
        #x = self.aggregator(x) #！[B,C,HW]
        #if not self.training:待恢复，看看crica
        #    x = self.relu(x)
        #    x = x.flatten(2).permute(0, 2, 1)#[B,HW,C]
        #    x = self.distill(x).view(B,H,W,C).permute(0, 3, 1, 2)#[B,C,H,W]
        #    x = self.aggregator(x)
        #    return x#, heatmaps

        B,C,HW = x.shape
        x_gem = x.view(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        x_gem = self.gem(x_gem)
        x_gem = self.fc(x_gem)
        x_gem = F.normalize(x_gem, p=2, dim=1)
        x = x.view(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))


        
        return x_gem,x
        

def seq_gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    B, D, SL = x.shape
    return F.avg_pool1d(x.clamp(min=eps).pow(p), SL).pow(1./p)


class SeqGeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        B, SL, D = x.shape
        x = einops.rearrange(x, "b sl d -> b d sl")
        x = seq_gem(x, p=self.p, eps=self.eps)
        assert x.shape == torch.Size([B, D, 1]), f"{x.shape}"
        return x[:, :, 0]
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class JistModelMe_seqvlad_with_final_norm_Me(nn.Module):
    def __init__(self, args, agg_type="seqvlad"):
        super().__init__()
        self.model = Me()
        self.seqvlad = SeqVLAD(seq_length=args.seq_length, dim=768,
                                     transf_backbone=False)
    

        
        self.features_dim = 768
        self.fc_output_dim = 768
        self.seq_length = args.seq_length
        self.aggregation_dim = self.fc_output_dim * 64

        
        self.agg_type = "seqvlad"
        
    def forward(self, x):
        return self.model(x)
    
    def aggregate(self, frames_features):
        aggregated_features = self.seqvlad(frames_features)
     
        return aggregated_features
        


