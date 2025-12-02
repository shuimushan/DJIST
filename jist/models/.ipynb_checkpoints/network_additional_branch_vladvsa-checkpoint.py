import torch
from torch import nn
import torch.nn.functional as F
import einops

import logging
import torchvision
from torch.nn.parameter import Parameter
from typing import Tuple
import copy

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

def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "EfficientNet_B0": 1280,
    "EfficientNet_B1": 1280,
    "EfficientNet_B2": 1408,
    "EfficientNet_B3": 1536,
    "EfficientNet_B4": 1792,
    "EfficientNet_B5": 2048,
    "EfficientNet_B6": 2304,
    "EfficientNet_B7": 2560,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int, train_all_layers : bool = False):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
            train_all_layers (bool): whether to freeze the first layers of the backbone during training or not.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone, train_all_layers)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str, train_all_layers : bool) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")

        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        if train_all_layers:
            logging.debug("Train all layers of the VGG-16")
        else:
            for layer in layers[:-5]:
                for p in layer.parameters():
                    p.requires_grad = False
            logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    elif backbone_name.startswith("EfficientNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.features.named_children():
                if name == "5": # Freeze layers before block 5
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only the last three blocks of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2] # Remove avg pooling and FC layer
    
    backbone = torch.nn.Sequential(*layers)
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim



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


class JistModel_additional_branch_vladvsa(nn.Module):
    def __init__(self, args, agg_type="concat"):
        super().__init__()
        
        self.model = GeoLocalizationNet(
    backbone=args.backbone,
    fc_output_dim=args.fc_output_dim,
    train_all_layers=True  # 或根据需求设置
)

        self._load_pretrained_weights('/mnt/JIST/weights/resnet18_512.pth')
        for name, param in self.model.named_parameters():
            if name.startswith("backbone.7"):  # Train only last residual block
                break
            param.requires_grad = False
        assert name.startswith("backbone.7"), "are you using a resnet? this only work with resnets"

        self.branch2 = self._get_copied_branch()
        
        
        self.features_dim = self.model.aggregation[3].in_features
        self.fc_output_dim = self.model.aggregation[3].out_features
        
        self.gem1_without_norm = GeMPool_without_norm()
        self.gem2_without_norm = GeMPool_without_norm()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
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
        
    def _load_pretrained_weights(self, weight_path):
        """加载预训练权重到backbone"""
        try:
            # 加载完整权重文件
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # 情况1：权重是完整的GeoLocalizationNet状态字典
            if 'backbone' in next(iter(checkpoint.keys())):
                msg = self.model.load_state_dict(checkpoint, strict=True)
                print("!1111") #是这种情况
            # 情况2：权重直接是backbone的状态字典
            else:
                msg = self.model.backbone.load_state_dict(checkpoint, strict=True)
            
            print(f"Loaded backbone weights from {weight_path}")
            #print(f"Missing keys: {msg.missing_keys}")
            #print(f"Unexpected keys: {msg.unexpected_keys}")
        
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
    

    def _get_copied_branch(self):
        """创建并复制最后两层的分支（不共享参数）"""
        layers = []
        for name, module in self.model.backbone.named_children():
            if name in ['7']:  
                # 深拷贝层结构和参数
                copied_layer = copy.deepcopy(module)
                # 重置参数梯度要求
                for param in copied_layer.parameters():
                    param.requires_grad = True
                layers.append(copied_layer)
        return nn.Sequential(*layers)
    
            
    def forward(self, x):
        for i in range(7):  # 前7层（0-6）
            x = self.model.backbone[i](x)
        branch1_feat = self.model.backbone[7](x)
        x_gem = self.model.aggregation(branch1_feat)
        
        branch2_feat = self.branch2(x)
        
        return x_gem,[branch1_feat,branch2_feat]
    
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