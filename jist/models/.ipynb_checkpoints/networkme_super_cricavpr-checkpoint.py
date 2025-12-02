
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
    def __init__(self, backbone_arch='dinov2_vitb14', pretrained=True, layer1=11, use_cls=False, norm_descs=True,out_indices=[8, 9, 10, 11],backbone_out_dim=3072,mix_in_dim=768,token_num=2,token_ratio=1,fc_output_dim=512):
        super().__init__()
        # get the backbone and the aggregator,先用默认的backbone参数
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1,  use_cls, norm_descs, out_indices)
        self.conv = nn.Conv2d(backbone_out_dim, mix_in_dim, (1, 1))#heatmap,维度待确定
        self.relu = nn.ReLU(inplace=True)
        if(token_num!=0):
            self.tokenmix = nn.Sequential(*[Tokenmixen(mix_in_dim, 16*16, token_ratio) for _ in range(token_num)])
        else:
            self.tokenmix = nn.Identity()
        self.gemfc = GeMFc(fc_output_dim=fc_output_dim)


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
        x_gem = self.gemfc(x_gem)

        
        return x_gem, x
        
        

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


class JistModelMe_super_cricavpr(nn.Module):
    def __init__(self, args, agg_type="concat"):
        super().__init__()
        self.model = Me(fc_output_dim=args.fc_output_dim)

        
        self.features_dim = 768
        self.fc_output_dim = args.fc_output_dim
        self.seq_length = args.seq_length
        if agg_type == "super_cricavpr":
            self.gem = helper.get_aggregator(agg_arch='GeM',agg_config={'p': 3})
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.aggregation_dim = 14*768
        else:
            print("wrong,only for super_cricavpr")
            exit()
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
        x = self.model(x)
        #x_1,x_2 =x#仅为了验证参数和FLOPs
        #x_2 = self.aggregate(x_2)#仅为了验证参数和FLOPs
        return x
    
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
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
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
            
            


