import torch
import torch.nn as nn
import timm
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F
from transformers import CLIPModel, CLIPVisionModel, AutoModel
from peft import LoraConfig, get_peft_model


class ClipModel(nn.Module):
    def __init__(self, fe_path=None, proto_path=None, 
                 feature_dim=1024, num_classes=1, freeze_backbone=True, device='cuda'):
        super(ClipModel, self).__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.foren_dim = 128
        
        cache_path = "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
        self.feature_extractor = CLIPModel.from_pretrained(cache_path, local_files_only=True)
        
        # If you have the access to huggingface, use this 
        # self.feature_extractor  = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        if freeze_backbone:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
            
        self.feature_extractor = self.feature_extractor.vision_model.to(device)
        self.foren_proj = nn.Linear(self.feature_dim, self.foren_dim, bias=False)
        self.fc = nn.Linear(self.foren_dim, num_classes, bias=False).to(self.device)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        full_sd = super(ClipModel, self).state_dict(destination=None, prefix=prefix, keep_vars=keep_vars)

        out = {} if destination is None else destination
        target_prefix = prefix + 'foren_proj.'
        for k, v in full_sd.items():
            if k.startswith(target_prefix):
                out[k] = v
        return out

    def forward(self, x):
        B = x.shape[0]
        x = self.feature_extractor(x)['pooler_output'] 
        x = self.foren_proj(x)
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        return x
    

    
class GAPLModel(nn.Module):
    def __init__(self, fe_path=None, proto_path=None, 
                 feature_dim=1024, num_classes=1, freeze_backbone=True, device='cuda'):
        super(GAPLModel, self).__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.foren_dim = 128
        self.n_prototype = 64
        cache_path = "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
        self.lora = not freeze_backbone
        self.feature_extractor = CLIPModel.from_pretrained(cache_path, local_files_only=True)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.foren_dim,
            num_heads=4,
            batch_first=True
        ).to(self.device)

        if freeze_backbone:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False

        if self.lora:
            self.peft = LoraConfig(
                task_type='FEATURE_EXTRACTION',
                r = 16,
                lora_alpha = 32,
                lora_dropout = 0.1,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            self.feature_extractor = get_peft_model(self.feature_extractor, self.peft)
            
        self.feature_extractor = self.feature_extractor.vision_model.to(device)
        self.proVec = None

        if proto_path is not None:
            proVec = torch.load(proto_path, map_location='cpu')
            self.proVec = proVec
            self.proVec = self.proVec.to(self.device)

        self.foren_proj = nn.Linear(self.feature_dim, self.foren_dim, bias=False)
        
        if fe_path is not None:
            fe_ckpt = torch.load(fe_path, map_location='cpu')
            ukeys, mkeys = self.load_state_dict(fe_ckpt['model'], strict=False)
            print(ukeys, mkeys)
 
        self.fc = nn.Linear(self.foren_dim, num_classes, bias=False).to(self.device)

    def forward(self, x, return_y=False):
        B = x.shape[0]
        x = self.feature_extractor(x)['pooler_output'] # [B, 1024]
        x = self.foren_proj(x)
        x = F.normalize(x, dim=1)

        pro = self.proVec.unsqueeze(0).expand(B, -1, -1)
        x, w = self.cross_attention(
            query = x.unsqueeze(1),
            key = pro,
            value = pro
        )
        x = self.fc(x.squeeze(1))

        if return_y:
            # return x, torch.cat([real_sim, fake_sim], dim=1)
            return x, w
        else:
            return x
    
    def load_prototype(self, x):
        self.proVec = torch.zeros_like(x)
        self.proVec = self.proVec.to(self.device)
        self.proVec.copy_(x)

if __name__ == '__main__':
    model = GAPLModel(fe_path='')
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad )
    print(f"Tunable parameter {num_param}")
    x = torch.randn(1, 3, 224, 224, device='cuda:0')
    y = model(x)
    print(y.shape)

    # state_dict = model.state_dict()
    # print(state_dict.keys())
