# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat
import copy


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2_self_additional_branch_only_last(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True, out_indices=[8, 9, 10, 11]) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.model_name = model_name.lower()  # 将大写转化为小写
        self.layer1 = layer1

        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name
        self.out_indices = out_indices


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为40层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load('./models/backbones/facebookresearch_dinov2_main/dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load('./weights/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为24层')
                exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load('./jist/models/backbones/facebookresearch_dinov2_main/dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load('./weights/dinov2_vitb14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2_vits14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vits14最高block层为12层')
                exit()
        else:
            print(f'模型名称定义错误，请检查model_name:{self.dino_model}是否正确')


        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)
                
        self.branch_block1 = copy.deepcopy(self.dino_model.blocks[10])
        self.branch_block2 = copy.deepcopy(self.dino_model.blocks[11])
            
        # 将新分支移到相同设备
        self.branch_block1 = self.branch_block1.to(self.device)
        self.branch_block2 = self.branch_block2.to(self.device)

        #self.dino_model.norm = nn.Sequential()
        #self.dino_model.head = nn.Sequential()


    def forward(self, x, masks=None):
        # 获取所有指定中间层的输出（序列格式）
        intermediate_outputs = self.dino_model.get_intermediate_layers(
            x, n=self.out_indices, reshape=False, return_class_token=True
        )
    
        # 提取特征序列（忽略class token）
        feature_sequences = [out[0] for out in intermediate_outputs]  # 每个元素是 [B, num_patches, C]
    
        # 将序列转换为特征图的函数
        def to_feature_map(patches):
            B, N, C = patches.shape
            H = W = int(N**0.5)  # 假设是正方形特征图
            return patches.permute(0, 2, 1).reshape(B, C, H, H)
    
        # === 主分支：原始模型的最后一层输出 ===
        # 使用原始模型的最后一层（out_indices[-1]）
        main_output = to_feature_map(feature_sequences[-1])
    
        # === 附加分支：使用分支block生成最后一层输出 ===
        # 从原始模型的第 out_indices[-3] 层开始（倒数第三层）
        branch_seq = feature_sequences[-3]  # 获取特征序列 [B, num_patches, C]
    
        # 通过两个分支block
        branch_layer1 = self.branch_block1(branch_seq)
        branch_output = self.branch_block2(branch_layer1)
    
        # 转换为特征图
        branch_output = to_feature_map(branch_output)
    
        return main_output, branch_output

# -----------------------------------------------debug-----------------------------------------------------------------/

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224).to('cuda')
    model = DinoV2_self(model_name='dinov2_vitb14', layer1=11, facet1="value", use_cls=False, norm_descs=True, device="cuda", pretrained=True)
    # torch.onnx.export(model.dino_model, torch.randn(1, 3, 224, 224), 'dinov2_vitl14.onnx', do_constant_folding=True, verbose=False)

    print(model)
    # print(model.dino_model.cls_token)
    # print(model.dino_model.pos_embed)
    # print(model.dino_model.mask_token)
    for name, param in model.dino_model.named_parameters():
        if param.requires_grad:
            print(f'***{name}**')

    print('-' * 70)
    summary(model, (3, 224, 224), 1, 'cuda')
    print('-' * 70)

    r = model(x)

    print_nb_params(model)

    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')


if __name__ == '__main__':
    main()


