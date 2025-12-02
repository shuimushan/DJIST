
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from . import helper
import math
import numpy as np
import einops

class GeMFc(nn.Module):
    """类似cosplace的顺序
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6,features_dim=768,fc_output_dim=512):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.fc = nn.Linear(features_dim, fc_output_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x

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

class Channelmixen(nn.Module):
    def __init__(self, fc_in_channels=768,
        in_channels=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(fc_in_channels),
            nn.Linear(fc_in_channels, in_channels),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(in_channels, fc_in_channels),
            #nn.Dropout(0.1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x + self.mlp(x)
        #Sprint(x.shape)
        return x




class Me(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, backbone_arch='dinov2_vitb14', pretrained=True, layer1=9, use_cls=False, norm_descs=True,out_indices=[11],backbone_out_dim=768,fc_output_dim=512):
        super().__init__()
        # get the backbone and the aggregator,先用默认的backbone参数
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1,  use_cls, norm_descs, out_indices)
        self.gemfc = GeMFc(fc_output_dim=fc_output_dim)


        #self.init_conv()

    #def init_conv(self):
    #    nn.init.constant_(self.conv.weight, 1.0 / 768)
    #    nn.init.zeros_(self.conv.bias)
    #    nn.init.constant_(self.conv_rev.weight, 1.0 / 768)
    #    nn.init.zeros_(self.conv_rev.bias)

    def forward(self, x):
        x = self.backbone(x)#B,C0,H,W    
        x = self.gemfc(x)

        
        return x
        
        

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


class JistModelMe_seqgem(nn.Module):
    def __init__(self, args, agg_type="concat"):
        super().__init__()
        self.model = Me(fc_output_dim = args.fc_output_dim)

        
        self.features_dim = 768
        self.fc_output_dim = args.fc_output_dim
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
            self.aggregation_dim = self.fc_output_dim
            self.seq_gem = SeqGeM()
        
        self.agg_type = agg_type
        
    def forward(self, x):
        x= self.model(x)
        return x,x
    
    def aggregate(self, frames_features):
        if self.agg_type == "concat":
            concat_features = einops.rearrange(frames_features, "(b sl) d -> b (sl d)", sl=self.seq_length)
            return concat_features
        if self.agg_type == "mean":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return aggregated_features.mean(1)
        if self.agg_type == "max":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return aggregated_features.max(1)[0]
        if self.agg_type == "conv1d":#原本的seqnet还有个avg_pool，但由于seqnet设置的sequential卷积核就是5，所以这里直接用的conv1d就够了
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
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return self.seq_gem(aggregated_features)


