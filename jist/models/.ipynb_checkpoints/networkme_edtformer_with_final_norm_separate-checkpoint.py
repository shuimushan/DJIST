
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import einops
import logging
import math
import torchvision


from .edtformer_backbone.backbone.vision_transformer import vit_base
from .edtformer_backbone.SACA import SA_CA

class EDTformer(nn.Module):#改了一下
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)

        self.fc = nn.Linear(768,768,bias=True)
        decoderlayer = SA_CA(d_model=768, nhead=16, batch_first=True)   # Simplified Decoder Block
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=2)

        # learnable queries
        self.queries = nn.Parameter(torch.zeros(1, 64, 768))
        nn.init.normal_(self.queries, std=1e-6)

        # linear projection for dimensionality adjustment
        self.channel_proj = nn.Linear(768, 256)
        self.row_proj = nn.Linear(64, 16)

    def forward(self, x):
        x = self.backbone(x)

        B,P,D = x["x_norm"].shape
        queries = self.queries.expand(B,-1,-1)

        x_c = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"]
        x_cp = torch.cat([x_c,x_p],dim=1)
        print(x_cp.shape)
        exit()

        x_cp2 = self.fc(x_cp)
        x = self.decoder(queries,x_cp2)
        x = self.channel_proj(x)
        x = self.row_proj(x.permute(0, 2, 1)).flatten(1)

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x_cp, x

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone



        
        

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


class JistModelMe_edtformer_with_final_norm_separate(nn.Module):
    def __init__(self, args, agg_type="concat"):
        super().__init__()
        self.model = EDTformer(pretrained_foundation = True, foundation_model_path = "/mnt/JIST/weights/dinov2_vitb14_pretrain.pth")

        
        self.features_dim = 768
        self.fc_output_dim = args.fc_output_dim#4096
        self.seq_length = args.seq_length

        if agg_type == "concat":
            self.aggregation_dim = self.fc_output_dim * args.seq_length
        if agg_type == "mean":
            self.aggregation_dim = self.fc_output_dim
        if agg_type == "max":
            self.aggregation_dim = self.fc_output_dim
        if agg_type == "conv1d":
            self.conv1d = torch.nn.Conv1d(self.fc_output_dim, self.fc_output_dim, self.seq_length)
            self.aggregation_dim = self.fc_output_dim
        if agg_type in ["simplefc", "meanfc"]:
            self.aggregation_dim = self.fc_output_dim
            self.final_fc = torch.nn.Linear(self.fc_output_dim * args.seq_length, self.fc_output_dim, bias=False)
        if agg_type == "meanfc":
            # Initialize as a mean pooling over the frames
            weights = torch.zeros_like(self.final_fc.weight)
            for i in range(self.fc_output_dim):
                for j in range(args.seq_length):
                    weights[i, j * self.fc_output_dim + i] = 1 / args.seq_length
            self.final_fc.weight = torch.nn.Parameter(weights)
        if agg_type == "seqgem":
            self.aggregation_dim = 4096#改了一下
            self.seq_gem = SeqGeM()
        
        self.agg_type = agg_type
        
    def forward(self, x):
        x_single,x_sequential = self.model(x)
        return x_single,x_sequential
    
    def aggregate(self, x):
        if self.agg_type == "concat":
            concat_features = einops.rearrange(frames_features, "(b sl) d -> b (sl d)", sl=self.seq_length)
            return concat_features
        if self.agg_type == "mean":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return aggregated_features.mean(1)
        if self.agg_type == "max":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return aggregated_features.max(1)[0]
        if self.agg_type == "conv1d":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            aggregated_features = einops.rearrange(aggregated_features, "b sl d -> b d sl", sl=self.seq_length)
            features = self.conv1d(aggregated_features)
            if len(features.shape) > 2 and features.shape[2] == 1:
                features = features[:, :, 0]
            return features
        if self.agg_type in ["simplefc", "meanfc"]:
            concat_features = einops.rearrange(frames_features, "(b sl) d -> b (sl d)", sl=self.seq_length)
            return self.final_fc(concat_features)
        if self.agg_type == "seqgem":
            aggregated_features = einops.rearrange(x, "(b sl) d -> b sl d", sl=self.seq_length)
            return F.normalize(self.seq_gem(aggregated_features), p=2, dim=-1)
        
        if self.agg_type == "super_cricavpr":
            B,C,HW = x.shape
            x = x.view(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
            x10,x11,x12,x13 = self.gem(x[:,:,0:8,0:8]),self.gem(x[:,:,0:8,8:]),self.gem(x[:,:,8:,0:8]),self.gem(x[:,:,8:,8:])
            x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.gem(x[:,:,0:5,0:5]),self.gem(x[:,:,0:5,5:11]),self.gem(x[:,:,0:5,11:]),\
                                        self.gem(x[:,:,5:11,0:5]),self.gem(x[:,:,5:11,5:11]),self.gem(x[:,:,5:11,11:]),\
                                        self.gem(x[:,:,11:,0:5]),self.gem(x[:,:,11:,5:11]),self.gem(x[:,:,11:,11:])
            x_crica = [i.unsqueeze(1) for i in [self.gem(x),x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]
            x_crica = torch.cat(x_crica,dim=1)
            x_crica = self.encoder(x_crica).view(B,14*C)
            x_crica = F.normalize(x_crica, p=2, dim=-1)
            #其实直接在这里把最后那个image取出来就好，这样连eval.py都不用改
            aggregated_features = einops.rearrange(x_crica, "(b sl) d -> b sl d", sl=self.seq_length)
            aggregated_features = aggregated_features[:,-1,:]
            return aggregated_features
            
            


