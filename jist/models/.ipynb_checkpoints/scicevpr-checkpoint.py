
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from . import helper
import math
import numpy as np

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




class SciceVPR(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, backbone_arch='dinov2_vitg14', pretrained=True, layer1=20, use_cls=False, norm_descs=True,out_indices=[8, 9, 10, 11],backbone_out_dim=1024,mix_in_dim=1024,token_num=1,token_ratio=1):
        super().__init__()
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1,  use_cls, norm_descs, out_indices)
        self.conv = nn.Conv2d(backbone_out_dim, mix_in_dim, (1, 1))#heatmap,维度待确定
        self.relu = nn.ReLU(inplace=True)
        self.tokenmix = nn.Sequential(*[Tokenmixen(mix_in_dim, 16*16, token_ratio) for _ in range(token_num)])
        self.gem = helper.get_aggregator(agg_arch='GeM',agg_config={'p': 3})
        encoder_layer = nn.TransformerEncoderLayer(d_model=mix_in_dim, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)#完全一致了
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


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
        x = x.view(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        x10,x11,x12,x13 = self.gem(x[:,:,0:8,0:8]),self.gem(x[:,:,0:8,8:]),self.gem(x[:,:,8:,0:8]),self.gem(x[:,:,8:,8:])
        x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.gem(x[:,:,0:5,0:5]),self.gem(x[:,:,0:5,5:11]),self.gem(x[:,:,0:5,11:]),\
                                        self.gem(x[:,:,5:11,0:5]),self.gem(x[:,:,5:11,5:11]),self.gem(x[:,:,5:11,11:]),\
                                        self.gem(x[:,:,11:,0:5]),self.gem(x[:,:,11:,5:11]),self.gem(x[:,:,11:,11:])
        x_crica = [i.unsqueeze(1) for i in [self.gem(x),x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]
        x_crica = torch.cat(x_crica,dim=1).view(B*14,C).unsqueeze(0)
        x_crica = self.encoder(x_crica).squeeze().view(B,14*C)
        x_crica = F.normalize(x_crica, p=2, dim=-1)
        
        return x_crica
        
        



