
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from . import helper
import math
import numpy as np
import einops

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
    
class GeMPool_without_norm(nn.Module):#这个gem pool的norm顺序跟crica不一样，不知道会不会有影响
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        #x = F.normalize(x, p=2, dim=1)
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        return x

class GeMFc(nn.Module):
    """类似cosplace的顺序
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6,features_dim=768,fc_output_dim=512):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.fc = nn.Linear(features_dim, fc_output_dim)
        self.gem = GeMPool()

    def forward(self, x):
        x_gem = self.gem(x)
        x_gem = self.fc(x_gem)
        x_gem = F.normalize(x_gem, p=2, dim=1)
        x_gem_no_fc = self.gem(x)
        x_gem_no_fc = F.normalize(x_gem_no_fc, p=2, dim=1)
        
        return x_gem,x_gem_no_fc

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
    def __init__(self, backbone_arch='additional_branch', pretrained=True, layer1=9, use_cls=False, norm_descs=True,out_indices=[8,9,10,11],backbone_out_dim=3072,mix_in_dim=768,token_num=2,token_ratio=1):
        super().__init__()
        # get the backbone and the aggregator,先用默认的backbone参数
        self.backbone = helper.get_backbone(backbone_arch, pretrained, 9,  use_cls, norm_descs, out_indices)
        self.conv = nn.Conv2d(backbone_out_dim, mix_in_dim, (1, 1))#heatmap,维度待确定
        self.relu = nn.ReLU(inplace=False)
        if(token_num!=0):
            self.tokenmix = nn.Sequential(*[Tokenmixen(mix_in_dim, 16*16, token_ratio) for _ in range(token_num)])
        else:
            self.tokenmix = nn.Identity()
        #self.gemfc = GeMFc()
        self.conv_addition = nn.Conv2d(backbone_out_dim, mix_in_dim, (1, 1))
        self.tokenmix_addition = nn.Sequential(*[Tokenmixen(mix_in_dim, 16*16, token_ratio) for _ in range(token_num)])
        self.gem = GeMPool()
        self.fc = nn.Linear(768, 512)


        #self.init_conv()

    #def init_conv(self):
    #    nn.init.constant_(self.conv.weight, 1.0 / 768)
    #    nn.init.zeros_(self.conv.bias)
    #    nn.init.constant_(self.conv_rev.weight, 1.0 / 768)
    #    nn.init.zeros_(self.conv_rev.bias)

    def forward(self, x):
        x,x_addition = self.backbone(x)#B,C0,H,W    
        x_addition = self.conv_addition(x_addition)
        x_addition = self.relu(x_addition).flatten(2)
        x_addition = self.tokenmix_addition(x_addition)
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
        x_addition = x_addition.view(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))

        
        return x_gem,[x,x_addition]
        
        

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


class R1_two_fc_with_final_norm_with_additional_branch_and_backbone_with_attention3_new(nn.Module):
    def __init__(self, args, agg_type="concat"):
        super().__init__()
        self.model = Me(layer1=args.layer1)
        print("layer1",args.layer1)

        self.scale = 768**-0.5
        self.features_dim = 768
        self.fc_output_dim = 512
        self.gem1_without_norm = GeMPool_without_norm()
        self.gem2_without_norm = GeMPool_without_norm()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(768, 512)
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
            self.aggregation_dim = self.fc_output_dim*2
            self.seq_gem1 = SeqGeM()
            self.seq_gem2 = SeqGeM()
        
        self.agg_type = agg_type
        
    def forward(self, x):
        return self.model(x)
    
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
           
            aggregated_features1 = frames_features[0]#b c h w
            aggregated_features1 = F.normalize(aggregated_features1, p=2, dim=1)#b c h w
            
            x_gem1 = self.gem1_without_norm(aggregated_features1)#b c
            x_gem1 = x_gem1.unsqueeze(dim=1)
            
            aggregated_features1 = einops.rearrange(aggregated_features1, "b c h w -> b (h w) c")# b 256 c
            
            atten1 = torch.matmul(x_gem1, aggregated_features1.transpose(1,2)).squeeze()#256
            #atten1 = atten1*self.scale
            #print(atten.shape)
            #atten1 = F.softmax(atten1, dim=-1).squeeze()#atten
            atten1 = F.normalize(atten1, p=2, dim=1)
            
            aggregated_features1 = x_gem1.squeeze()
            aggregated_features1 = F.normalize(aggregated_features1, p=2, dim=1)
            aggregated_features1 = einops.rearrange(aggregated_features1, "(b sl) d -> b sl d", sl=self.seq_length)
            aggregated_features1 = self.fc1(self.seq_gem1(aggregated_features1))
            aggregated_features1 = F.normalize(aggregated_features1, p=2, dim=1)
            
           
            aggregated_features2 = frames_features[1]#b c h w
            aggregated_features2 = F.normalize(aggregated_features2, p=2, dim=1)#b c h w
            
            x_gem2 = self.gem2_without_norm(aggregated_features2)#b c
            x_gem2 = x_gem2.unsqueeze(dim=1)
            
            aggregated_features2 = einops.rearrange(aggregated_features2, "b c h w -> b (h w) c")# b 256 c
            
            atten2 = torch.matmul(x_gem2, aggregated_features2.transpose(1,2)).squeeze()#256
            #atten2 = atten2*self.scale
            #print(atten.shape)
            #atten2 = F.softmax(atten2, dim=-1).squeeze()#atten
            atten2 = F.normalize(atten2, p=2, dim=1)
            
            aggregated_features2 = x_gem2.squeeze()
            aggregated_features2 = F.normalize(aggregated_features2, p=2, dim=1)
            aggregated_features2 = einops.rearrange(aggregated_features2, "(b sl) d -> b sl d", sl=self.seq_length)
            aggregated_features2 = self.fc2(self.seq_gem2(aggregated_features2))
            aggregated_features2 = F.normalize(aggregated_features2, p=2, dim=1)
            aggregated_features = torch.cat([aggregated_features1,aggregated_features2],dim=1)
            return [aggregated_features,atten1,atten2]


