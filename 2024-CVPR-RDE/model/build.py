from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .prototype import PrototypeBranch
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        self.prototype_enabled = (
            getattr(args, "prototype", False)
            or getattr(args, "use_loss_id", False)
        )
        self.prototype_feature_source = self._resolve_prototype_feature_source()
        if self.prototype_enabled:
            image_dim, text_dim = self._prototype_feature_dims()
            self.prototype_branch = PrototypeBranch(args, num_classes, image_dim=image_dim, text_dim=text_dim)
        else:
            self.prototype_branch = None
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [
            l.strip()
            for l in loss_names.split('+')
            if l.strip() and l.strip().lower() != "proto"
        ]
        print(f'Training Model with {self.current_task} tasks')

    def _resolve_prototype_feature_source(self):
        feature = getattr(self.args, "prototype_feature", "auto")
        if feature == "auto":
            return "global"
        if feature == "local":
            return "tse"
        if feature in ("global", "tse"):
            return feature
        raise ValueError(f"Unknown --prototype_feature: {feature}")

    def _prototype_feature_dims(self):
        if self.prototype_feature_source == "tse":
            return self.visul_emb_layer.embed_dim, self.texual_emb_layer.embed_dim
        return self.embed_dim, self.embed_dim
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def _compute_host_embeddings(self, images, caption_ids):
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        return {
            "i_feats": i_feats,
            "t_feats": t_feats,
            "i_tse_f": i_tse_f.float(),
            "t_tse_f": t_tse_f.float(),
        }

    def select_prototype_features(self, outputs, batch=None):
        if self.prototype_feature_source == "tse":
            return outputs["i_tse_f"], outputs["t_tse_f"]
        return outputs["i_feats"], outputs["t_feats"]

    @torch.no_grad()
    def extract_prototype_features(self, batch):
        outputs = self._compute_host_embeddings(batch["images"], batch["caption_ids"])
        return self.select_prototype_features(outputs, batch)

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        outputs = self._compute_host_embeddings(images, caption_ids)
        i_feats = outputs["i_feats"]
        t_feats = outputs["t_feats"]
        i_tse_f = outputs["i_tse_f"]
        t_tse_f = outputs["t_tse_f"]

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        outputs = self._compute_host_embeddings(images, caption_ids)
        i_feats = outputs["i_feats"]
        t_feats = outputs["t_feats"]
        i_tse_f = outputs["i_tse_f"]
        t_tse_f = outputs["t_tse_f"]
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        proto_image_feats = None
        proto_text_feats = None
        if getattr(self.args, "track_train_diagnostics", True) or self.prototype_branch is not None:
            proto_image_feats, proto_text_feats = self.select_prototype_features(outputs, batch)

        if getattr(self.args, "track_train_diagnostics", True):
            ret["_diag"] = {
                "host_image_feats": proto_image_feats.detach(),
                "host_text_feats": proto_text_feats.detach(),
                "proto_image_feats": proto_image_feats.detach(),
                "proto_text_feats": proto_text_feats.detach(),
                "pids": batch["pids"].detach(),
                "indices": batch.get("index", None),
            }

        if self.prototype_branch is not None:
            proto_ret = self.prototype_branch(
                proto_image_feats,
                proto_text_feats,
                batch['pids'],
                use_loss_id=getattr(self.args, "use_loss_id", False),
            )
            if "proto_id_loss" in proto_ret:
                ret["proto_id_loss"] = proto_ret["proto_id_loss"] * getattr(self.args, "prototype_id_weight", 0.2)
  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    if getattr(model, "prototype_branch", None) is not None:
        model.prototype_branch.float()
    return model
