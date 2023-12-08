from code import interact
import os
from collections import OrderedDict
from random import shuffle
from types import SimpleNamespace
import torch
from torch import nn
from torch._C import device
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME, tokenize
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, Slip
from .transformer import DualTransformer
from .transformer.mutihead_attention import MultiheadAttention
from .transformer.xpool import XPool
from .loss import ivc_loss, cal_nll_loss, rec_loss
from .PDE import DisTrans
from .loss import compute_dis_contrast
import numpy as np
allgather = AllGather.apply
allgather2 = AllGather2.apply
import pdb
import math
import itertools
# from .eva_clip import build_eva_model_and_transforms
class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())            
            dim = middle_dim
            layers.append(nn.Sequential(*block))        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)
    
class SLIP(nn.Module):
    def __init__(self, config):
        super(SLIP, self).__init__()
        # eva_clip_path = "/mnt/bd/cxx-dataset/PMTVR/eva_clip_psz14.pt" # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt
        # model_name = "EVA_CLIP_g_14"
        # model, preprocess = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)

        self.config = config
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.embd_mode = config.embd_mode
        if self.embd_mode == 'emcl':
            self.slip = Slip(k=config.centerK,
                            stage_num=config.stage_num,
                            momentum=config.momentum,
                            lamd=config.lamd,
                            beta=config.beta)

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        # self.cross_attn_ = MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        
        # self.attn = MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        ## embedding mode
        self.dist_video_trans = DisTrans(transformer_width, transformer_heads)
        self.dist_text_trans = DisTrans(transformer_width, transformer_heads)

        self.interact_mode = config.interact_mode
        self.num_props = config.num_props
        if self.embd_mode == 'xpool':
            self.xpool = XPool(config)
        self.dist_text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.dist_video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.v2w_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))

        # self.text_saliency_fc = nn.Sequential(
        #     nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
        #     nn.Linear(transformer_width, config.max_words))
        # self.video_saliency_fc = nn.Sequential(
        #     nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
        #     nn.Linear(transformer_width, config.max_frames))
        # self.temporal_order_fc = nn.Sequential(
        #     nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
        #     nn.Linear(transformer_width, config.max_frames))

        # self.moment_fc = nn.Sequential(
            # nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            # nn.Linear(transformer_width, self.num_props * 2))

        ## ===> generate Gaussian masks
        self.mse_loss = nn.MSELoss(reduction='none')
        self.rec_loss = True
        self.do_gauss = config.do_gauss

        
        self.dropout = 0.1
        self.sal_pred = config.sal_predictor
        if 'a' in self.sal_pred:
            self.attn = nn.MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        # self.saliency_video_trans = nn.Transformer(embed_dim, transformer_heads//2, num_decoder_layers=config.sal_trans_num_layers, num_encoder_layers=config.sal_trans_num_layers, dim_feedforward= embed_dim << 1)
        # self.saliency_text_trans = nn.Transformer(embed_dim, transformer_heads//2, num_decoder_layers=config.sal_trans_num_layers, num_encoder_layers=config.sal_trans_num_layers, dim_feedforward= embed_dim << 1)
        # self.rec_video_trans1 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers1, num_decoder_layers2=config.rec_trans_num_layers1)
        # self.rec_text_trans1 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers2, num_decoder_layers2=config.rec_trans_num_layers2)
        # self.rec_video_trans2 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers1, num_decoder_layers2=config.rec_trans_num_layers1)
        # self.rec_text_trans2 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers2, num_decoder_layers2=config.rec_trans_num_layers2)
        # self.temporal_trans = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.tmp_trans_num_layers, num_decoder_layers2=config.tmp_trans_num_layers)
        # self.video_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        # self.text_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        # self.fc_gauss = nn.Linear(embed_dim, self.num_props*2)
        self.sigma = config.sigma
        self.gamma = config.gamma
        # self.word_fc = nn.Linear(embed_dim, embed_dim)
        # self.fc_comp = nn.Linear(embed_dim, vocab_size)
        # self.mask_video_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        # self.mask_text_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        self.max_epoch = config.epochs
        self.use_negative = True
        self.temp_loss_weight = config.temp_loss_weight
        self.rec_loss_weight = config.rec_loss_weight
        self.ret_weight = config.ret_weight
        self.pseudo_ret_weight = config.pseudo_ret_weight
        self.ot_weight = config.ot_weight
        # self.trans = DualTransformer()
        self.sample_num = config.sample_num
        self.distribution = config.distribution
        # self.dis_fc1 = nn.Linear(embed_dim * self.sample_num, embed_dim)
        # self.dis_fc2 = nn.Linear(embed_dim * self.sample_num, embed_dim)
        ## ===> end of generate Gaussian masks
        self.shuffle_idx = None
        self.unshuffle_idx = None

        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config
            
        if self.agg_module in ["seqLSTM", "seqTransf","seq"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers//2,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        
        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        if config.freeze_clip:
            print("Turning off gradients in both the video and the text encoder")
            for name, param in self.clip.named_parameters():
                    param.requires_grad_(False)
        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()
                
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        ## <=== End of initialization trick

        self.eps = 0.1
        self.max_iter = 100
        self.negative_scale=self.config.negative_scale #1/2000
        self.shift=self.config.shift # 0.1

        self.img2txt = IM2TEXT(transformer_width, transformer_width, transformer_width)

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):

        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, pool_text_feat, text_mask, video_mask, pseudo_text, pool_pseudo_text, pseudo_text_mask = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True, gauss=self.do_gauss)

        if self.training:
            if torch.cuda.is_available():  # batch merge here
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                pool_text_feat = allgather(pool_text_feat, self.config)
                pseudo_text = allgather(pseudo_text, self.config)
                pool_pseudo_text = allgather(pool_pseudo_text, self.config)
                pseudo_text_mask = allgather(pseudo_text_mask, self.config)
                torch.distributed.barrier()  # force sync

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            
            # loss = 0.
            tv_logits, pv_logits, tp_logits = self.get_similarity_logits(text_feat, pool_text_feat, text_mask, video_feat, video_mask, pseudo_text, pool_pseudo_text, pseudo_text_mask)
            logit_scale = self.clip.logit_scale.exp()

            tv_loss_t2v = self.loss_fct(tv_logits * logit_scale)
            tv_loss_v2t = self.loss_fct(tv_logits.T * logit_scale)        
            tv_loss = (tv_loss_t2v + tv_loss_v2t) / 2

            pv_loss_p2v = self.loss_fct(pv_logits * logit_scale)
            pv_loss_v2p = self.loss_fct(pv_logits.T * logit_scale)
            pv_loss = (pv_loss_p2v + pv_loss_v2p) / 2

            cos_criterion = nn.CosineEmbeddingLoss()
            # target = torch.ones(pseudo_text)
            target = torch.as_tensor([1]).to(text_feat.device)
            tp_loss = cos_criterion(pool_text_feat, pool_pseudo_text, target)
            # tp_loss_t2p = self.loss_fct(tp_logits * logit_scale)
            # tp_loss_p2t = self.loss_fct(tp_logits.T * logit_scale)
            # tp_loss = (tp_loss_t2p + tp_loss_p2t) / 2
            

            # retrieval_loss, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, dis_text_feat, dis_video_feat = self.get_similarity_loss(text_feat, pool_text_feat, video_feat, text_mask, video_mask,shaped=True,v2w=False)
            # pseudo_retrieval_loss1, *_ = self.get_similarity_loss(pseudo_text, pool_pseudo_text, video_feat, pseudo_text_mask, video_mask,shaped=True,v2w=True)

            # pool_pseudo_text = pool_pseudo_text/pool_pseudo_text.norm(dim=-1, keepdim=True)
            # pool_text_feat = pool_text_feat/pool_text_feat.norm(dim=-1, keepdim=True)

            # sim = torch.einsum("ad,bd->ab",pool_pseudo_text, pool_text_feat)
            # logit_scale = self.clip.logit_scale.exp()

            # loss_1 = self.loss_fct(sim * logit_scale)
            # loss_2 = self.loss_fct(sim.T * logit_scale)
            
            # pseudo_retrieval_loss2 = (loss_1 + loss_2) / 2

            if self.distribution:
                
                # print("w/o dist loss")
                dis_cl_loss = compute_dis_contrast(txt_mu, torch.exp(txt_logsigma), vid_mu, torch.exp(vid_logsigma),self.negative_scale,self.shift)
                final_loss = retrieval_loss*self.ret_weight + dis_cl_loss*self.dist_weight
                final_loss_dict = {'final_loss': final_loss.item(), 
                                'retrieval_loss': retrieval_loss.item()*self.ret_weight, 
                                'dis_cl_loss': dis_cl_loss.item()*self.dist_weight, 
                                }
            else:
                final_loss = self.ret_weight * tv_loss + pv_loss + tp_loss
                # final_loss = pv_loss
                final_loss_dict = {'final_loss': final_loss.item(), 
                                'tv_loss': tv_loss.item()*self.ret_weight, 
                                'pv_loss': pv_loss.item()*self.pseudo_ret_weight, 
                                'tp_loss': tp_loss.item()*self.pseudo_ret_weight, 
                                }
            
            return final_loss, final_loss_dict
        else:
            return None
            
    def get_moment_text_rec(self, text_feat, video_feat, text_mask, video_mask, props, text_weight):
        bsz, frame_len, T = video_feat.shape
        props = torch.sigmoid(props).view(bsz*self.num_props, 2)
        gauss_center = props[:, 0]
        gauss_width = props[:, 1]

        video_feat = video_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, -1, T)
        video_mask = video_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)    
        
        masked_text, masked_vec_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=self.config.text_mask_rate)
        text_feat = text_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, -1, T)
        text_mask = text_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)      
        masked_text = masked_text.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, -1, T)
        masked_vec_text = masked_vec_text.squeeze().unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        text_weight = text_weight.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)        

        gauss_weight = self.generate_gauss_weight(frame_len, gauss_center, gauss_width)
        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        mask_moment, masked_vec_video = self._mask_moment(video_feat, video_mask, gauss_center, gauss_width)

        rec_text = self.rec_text_trans2(video_feat, None, masked_text, None, decoding=2, gauss_weight=pos_weight)[1]
        rec_video = self.rec_video_trans2(text_feat, None, mask_moment, None,  decoding=2, gauss_weight=text_weight)[1]

        rec_video_loss = self.mse_loss(rec_video, video_feat)
        rec_text_loss = self.mse_loss(rec_text, text_feat)

        rec_video_loss = rec_video_loss * masked_vec_video * pos_weight.unsqueeze(2)
        rec_text_loss = rec_text_loss * masked_vec_text.unsqueeze(-1) * text_weight.unsqueeze(2)
        return rec_text_loss.mean(), rec_video_loss.mean()

    def BCE_loss(self, logits, labels, mask):
        labels = labels.type_as(logits)
        loss_per_location = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        mask = mask.type_as(logits)
        loss = (loss_per_location * mask).sum() / (mask.sum() + 1e-4)
        return loss

    def get_temporal_order_loss(self, text_feat, video_feat, text_mask, video_mask, text_weight, video_weight):
        B, T, D = video_feat.shape
        shuffle_idx = torch.from_numpy(np.random.permutation(np.arange(T))).to(video_feat.device)
        shuffle_video_feat = video_feat[:,shuffle_idx]
        shuffle_video_mask = video_mask[:,shuffle_idx]
        recover_idx = torch.argsort(shuffle_idx)
        shuffle_out = self.temporal_trans(text_feat, None, shuffle_video_feat, None,  decoding=2, gauss_weight=text_weight)[1]
        # shuffle_out = self.temporal_trans(text_feat, text_mask, shuffle_video_feat, shuffle_video_mask,  decoding=2, gauss_weight=text_weight)[1]
        temporal_order_pred = self.temporal_order_fc(shuffle_out).squeeze()
        shuffle_idx = shuffle_idx.unsqueeze(0).repeat(B,1)
        # temporal_loss = self.BCE_loss(temporal_order_pred, shuffle_idx, shuffle_video_mask)
        # temporal_loss = nn.BCEWithLogitsLoss(pos_weight=shuffle_video_mask)(temporal_order_pred, shuffle_idx.float())
        # temporal_loss = nn.CrossEntropyLoss()(temporal_order_pred, shuffle_idx)
        temporal_loss = self.nll_loss(temporal_order_pred, shuffle_idx, shuffle_video_mask, video_weight).mean()
        return temporal_loss

    def nll_loss(self, logit, idx, mask, weights=None):
        eps = 0.1
        acc = (logit.max(dim=-1)[1]==idx).float()
        mean_acc = (acc * mask).sum() / mask.sum()
        # logit = logit.log_softmax(dim=-1).log_softmax(dim=-2)
        logit = logit.log_softmax(dim=-1)
        nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -logit.sum(dim=-1)
        nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss

        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        if weights is not None:
            nll_loss = nll_loss * weights

        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
        # nll_loss = (nll_loss * weights).sum(dim=-1)

        return nll_loss

    def get_rec_loss(self, text_feat, video_feat, text_mask, video_mask, text_weight, video_weight):
        # text_weight = allgather(text_weight, self.config)
        # video_weight = allgather(video_weight, self.config)

        # random mask_rete 
        # text_mask_rate = np.random.uniform(0, 1.0)
        # video_mask_rate = np.random.uniform(0, 1.0)
        # masked_video = self._mask_feat(video_feat, video_mask.sum(1), video_weight, mask_rate=video_mask_rate )
        # masked_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=text_mask_rate)

        # mask_rete 
        masked_video, masked_vec_video = self._mask_feat(video_feat, video_mask.sum(1), video_weight, mask_rate=self.config.video_mask_rate)
        masked_text, masked_vec_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=self.config.text_mask_rate)

        # #  p = random
        # masked_video = self._mask_feat(video_feat, video_mask.sum(1), mask_rate=self.config.video_mask_rate)
        # masked_text = self._mask_feat(text_feat, text_mask.sum(1), mask_rate=self.config.text_mask_rate)
        # w/ mask
        # rec_video = self.rec_trans(masked_video, video_mask, text_feat, text_mask, decoding=1, gauss_weight=text_weight)[1]
        # rec_text = self.rec_trans(video_feat, video_mask, masked_text, text_mask, decoding=2, gauss_weight=video_weight)[1]

        rec_video = self.rec_video_trans1(text_feat, None, masked_video, None,  decoding=2, gauss_weight=text_weight)[1]
        rec_text = self.rec_text_trans1(video_feat, None, masked_text, None, decoding=2, gauss_weight=video_weight)[1]

        # w/o gauss weight
        # rec_video = self.rec_video_trans(text_feat, None, masked_video, None,  decoding=2, gauss_weight=text_weight)[1]
        # rec_text = self.rec_text_trans(video_feat, None, masked_text, None, decoding=2, gauss_weight=video_weight)[1]

        rec_video_loss = self.mse_loss(rec_video, video_feat)
        rec_text_loss = self.mse_loss(rec_text, text_feat)

        rec_video_loss = rec_video_loss * masked_vec_video * video_weight.unsqueeze(2)
        rec_text_loss = rec_text_loss * masked_vec_text * text_weight.unsqueeze(2)
        return rec_video_loss.mean(), rec_text_loss.mean()

    def Wasserstein2(self, mu1, sigma1, mu2, sigma2): # 2W距离，传入图片和文本的均值和标准差
        bs1 = mu1.shape[0]
        bs2 = mu2.shape[0]
        mu1 = torch.stack([mu1]*bs2, dim=1)
        sigma1 = torch.stack([sigma1]*bs2, dim=1)
        mu2 = torch.stack([mu2]*bs1, dim=0)
        sigma2 = torch.stack([sigma2]*bs1, dim=0)
        p1 = torch.sum(torch.pow(mu1 - mu2, 2), dim=-1)
        p2 = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=-1)
        return p1+p2, p1
    
    def get_wti(self, text_feat, text_mask, text_weight, video_feat, video_mask, video_weight):

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0   

        return retrieve_logits

    def get_similarity_logits(self, text_feat, pool_text_feat, text_mask, video_feat, video_mask, pseudo_text, pool_pseudo_text, pseudo_text_mask):
        pseudo_text_weight = self.v2w_weight_fc(pseudo_text).squeeze(2)  # B_t x N_t x D -> B_t x N_t
        text_weight = self.dist_text_weight_fc(text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
        video_weight = self.dist_video_weight_fc(video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v

        pseudo_text_weight.masked_fill_(torch.tensor((1 - pseudo_text_mask), dtype=torch.bool), float("-inf"))
        pseudo_text_weight = torch.softmax(pseudo_text_weight, dim=-1)  # B_t x N_t

        text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
        # text_weight = torch.sigmoid(text_weight)  # B_t x N_t            

        video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B_v x N_v
        # video_weight = torch.sigmoid(video_weight)  # B_v x N_v
        
        pseudo_text = pseudo_text / pseudo_text.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)         

        tv_logits = self.get_wti(text_feat, text_mask, text_weight, video_feat, video_mask, video_weight)
        # video_feat = self._mean_pooling_for_similarity_visual(video_feat, video_mask)
        # video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) 
        # pv_logits = 
        pv_logits = self.get_wti(pseudo_text, pseudo_text_mask, pseudo_text_weight, video_feat, video_mask, video_weight)
        # tp_logits = None
        tp_logits = self.get_wti(text_feat, text_mask, text_weight, pseudo_text, pseudo_text_mask, pseudo_text_weight)
        # if self.training:
        #     return tv_logits, pv_logits, tp_logits
        return tv_logits, pv_logits, tp_logits

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def get_vis(self, text_feat, video_feat):
        import numpy as np
        from sklearn.manifold import TSNE
        from sklearn.cluster import DBSCAN
        import matplotlib.pyplot as plt
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        # 假设 video_feat 是视频特征的 PyTorch tensor，text_feat 是文本特征的 PyTorch tensor
        # video_feat 的形状为 [32, 12, 512]
        # text_feat 的形状为 [32, 32, 512]

        # 将 PyTorch tensor 转换为 NumPy 数组
        text_feat = text_feat.unsqueeze(1)
        video_feat = video_feat.unsqueeze(1)
        video_feat = video_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        batch_size, num_frames, feature_dim = video_feat.shape
        _, num_words, _ = text_feat.shape

        # 使用 t-SNE 分别降维视频特征和文本特征
        tsne_video = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_text = TSNE(n_components=2, perplexity=30, n_iter=300)

        # 创建颜色列表，为每个 batch_size 分配不同的颜色
        colors = plt.cm.get_cmap('tab20', batch_size)  # 使用tab20色表，可根据 batch_size 调整
        # 
        # 绘制视频特征的 t-SNE 图
        for i in range(batch_size):
            tsne_result_video = tsne_video.fit_transform(video_feat[i])
            color = colors(i)  # 为每个 batch_size 分配不同的颜色
            for j in range(num_frames):
                plt.scatter(tsne_result_video[j, 0], tsne_result_video[j, 1], c=color, marker='^')

        # 绘制文本特征的 t-SNE 图
        for i in range(batch_size):
            tsne_result_text = tsne_text.fit_transform(text_feat[i])
            color = colors(i)  # 为每个 batch_size 分配不同的颜色
            for j in range(num_words):
                plt.scatter(tsne_result_text[j, 0], tsne_result_text[j, 1], c=color, marker='o')

        # 绘制特征边界
        for i in range(batch_size):
            unique_labels_video = np.unique(DBSCAN(eps=1.0, min_samples=5).fit(tsne_result_video).labels_)
            unique_labels_text = np.unique(DBSCAN(eps=1.0, min_samples=5).fit(tsne_result_text).labels_)

            for label in unique_labels_video:
                if label == -1:
                    continue  # 跳过噪声点
                cluster_points = tsne_result_video[DBSCAN(eps=1.0, min_samples=5).fit(tsne_result_video).labels_ == label]
                plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'k-', alpha=0.3)

            for label in unique_labels_text:        
                if label == -1:
                    continue  # 跳过噪声点
                cluster_points = tsne_result_text[DBSCAN(eps=1.0, min_samples=5).fit(tsne_result_text).labels_ == label]
                plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'k-', alpha=0.3)

        plt.title("t-SNE Visualization with Feature Boundaries")

        plt.savefig("vis.jpg")

    def get_dist_vis(self, text_feat, txt_mu, txt_logsigma, video_feat, vid_mu, vid_logsigma):
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        # text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        # 假设 video_feat 是视频特征的 PyTorch tensor，text_feat 是文本特征的 PyTorch tensor
        # video_feat 的形状为 [32, 12, 512]
        # text_feat 的形状为 [32, 32, 512]

        # 将 PyTorch tensor 转换为 NumPy 数组
        # video_feat = video_feat.cpu().numpy()
        # text_feat = text_feat.cpu().numpy()

        # 假设 video_feat_mean、video_feat_logsigma、text_feat_mean、text_feat_logsigma
        # 分别是视频特征的均值、对数标准差，以及文本特征的均值和对数标准差
        # 形状为 [batch_size, num_frame, dim] 或 [batch_size, num_words, dim]

        batch_size, num_frames, dim = vid_mu.shape

        # 创建一个新的图
        fig, ax = plt.subplots()

        # 遍历每个视频或文本的均值和对数标准差特征
        for i in range(batch_size):
            # 画视频特征的椭圆
            video_mean_feat = vid_mu[i]  # 视频特征的均值
            video_logsigma_feat = vid_logsigma[i]  # 视频特征的对数标准差

            # 计算协方差矩阵，假设特征之间是独立的
            video_cov_matrix = np.diag(np.exp(2 * video_logsigma_feat))

            # 获取视频协方差矩阵的对角线元素
            if video_cov_matrix.ndim == 2:
                video_std_dev = np.sqrt(np.diagonal(video_cov_matrix))
            elif video_cov_matrix.ndim == 3:
                video_std_dev = np.sqrt(np.diagonal(video_cov_matrix, axis1=1, axis2=2))
            else:
                video_std_dev = np.sqrt(np.diagonal(video_logsigma_feat))

            # 创建视频特征的椭圆
            video_ellipse = Ellipse(video_mean_feat, width=2 * video_std_dev, height=2 * video_std_dev, edgecolor='b', fc='None', lw=2)

            # 将视频特征的椭圆添加到图中
            ax.add_patch(video_ellipse)


            # 画文本特征的椭圆
            text_mean_feat = txt_mu[i]  # 文本特征的均值
            text_logsigma_feat = txt_logsigma[i]  # 文本特征的对数标准差

            # 计算协方差矩阵，假设特征之间是独立的
            text_cov_matrix = np.diag(np.exp(2 * text_logsigma_feat))

            # 获取文本协方差矩阵的对角线元素
            if text_cov_matrix.ndim == 2:
                text_std_dev = np.sqrt(np.diagonal(text_cov_matrix))
            elif text_cov_matrix.ndim == 3:
                text_std_dev = np.sqrt(np.diagonal(text_cov_matrix, axis1=1, axis2=2))
            else:
                text_std_dev = np.sqrt(np.diagonal(text_logsigma_feat))

            # 创建文本特征的椭圆
            text_ellipse = Ellipse(text_mean_feat, width=2 * text_std_dev, height=2 * text_std_dev, edgecolor='r', fc='None', lw=2)

            # 将文本特征的椭圆添加到图中
            ax.add_patch(text_ellipse)

        # 设置坐标轴的范围
        ax.set_xlim(-5, 5)  # 根据你的数据范围来设置
        ax.set_ylim(-5, 5)  # 根据你的数据范围来设置

        # 添加标题和标签
        plt.title("Feature Distribution with Mean and Log Sigma")
        plt.xlabel("Feature Dimension 1")
        plt.ylabel("Feature Dimension 2")

        # 显示图
        # plt.show()
        plt.savefig("dist_vis.jpg")

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
            mu.unsqueeze(1))
        return samples

    def _mean_pooling_for_similarity_visual(self, video_feat, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        if video_feat.dim()>3:
            video_mask_un = video_mask_un.unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_out
    def _mean_pooling_for_similarity_sequence(self, text_feat, text_mask):
        text_mask_un = text_mask.to(dtype=torch.float).unsqueeze(-1)
        if text_feat.dim()>3:
            text_mask_un = text_mask_un.unsqueeze(-1)
        text_mask_un[:, 0, :] = 0.
        text_feat = text_feat * text_mask_un
        text_out = torch.sum(text_feat, dim=1) / torch.sum(text_mask_un, dim=1, dtype=torch.float)
        return text_out

    def get_text_feat(self, text_ids, text_mask, shaped=False, gauss=False, token_type=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)

        if gauss:
            text_vec = self.text_vec.view(1, 1, -1).expand(bs_pair, 1, -1).float() #
            # text_ids = torch.cat([text_ids, text_vec], dim=1) #
            mask = torch.zeros(bs_pair).byte().cuda().view(-1,1)
            text_mask = torch.cat([text_mask, mask], dim=1) #
            pool_text_feat, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask, text_vec=text_vec)
        else:
            # 
            pool_text_feat, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask, token_type=token_type)
        # pool_text_feat, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)

        pool_text_feat, text_feat = pool_text_feat.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        pool_text_feat = pool_text_feat.view(bs_pair, -1, pool_text_feat.size(-1)).squeeze(1)
        return text_feat, pool_text_feat, text_mask

    def get_video_feat(self, video, video_mask, shaped=False, gauss=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
        
        bs_pair, n_v = video_mask.size()
        image_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        image_feat = image_feat.float().view(bs_pair, -1, image_feat.size(-1))

        # add Gaussian masks
        if gauss:
            video_vec = self.video_vec.view(1, 1, -1).expand(bs_pair, 1, -1) #
            image_feat = torch.cat([image_feat, video_vec], dim=1) #
            mask = torch.zeros(bs_pair).byte().cuda().view(-1,1)
            image_feat = torch.cat([video_mask, mask], dim=1) #
        video_feat = self.agg_video_feat(image_feat, video_mask, self.agg_module)
        # TODO: video2text or image2text
        return video_feat, video_feat, video_mask

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False, gauss=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
        video_feat, image_feat, video_mask = self.get_video_feat(video, video_mask, shaped=True, gauss=gauss)
        # image_feat += torch.rand(image_feat.shape[0], device=image_feat.device).view(-1,1,1)*torch.randn(image_feat.shape, device=image_feat.device)
        v2w_feat = self.img2txt(image_feat).float()
        # v2w_feat += torch.rand(v2w_feat.shape[0], device=v2w_feat.device).view(-1,1,1)*torch.randn(v2w_feat.shape, device=v2w_feat.device)
        # pdb.set_trace()
        # pseudo_text, pool_pseudo_text,  pseudo_text_mask = self.get_text_feat(v2w_feat, video_mask, shaped=True, gauss=gauss, token_type=True)
        prompt = tokenize("a video of").type(text_ids.dtype)
        prompt = prompt[:,:5]

        prompt = prompt.to(v2w_feat.device)
        prompt = prompt.view(1, -1)
        prompt = prompt.repeat(v2w_feat.size(0), 1)

        pseudo_mask = torch.ones(prompt.size()[0],5).to(v2w_feat.device)
        # text_features = self.clip.encode_text_img(prompt, v2w_feat)
        # pool_pseudo_text, pseudo_text,  pseudo_text_mask= self.clip.img2text(prompt, v2w_feat, return_hidden=True, mask=pseudo_mask)
        pool_pseudo_text, pseudo_text,  pseudo_text_mask= self.clip.img2text(text_ids, v2w_feat, return_hidden=True, mask=text_mask)
        text_feat, pool_text_feat, text_mask = self.get_text_feat(text_ids, text_mask, shaped=True, gauss=gauss)
        
        return text_feat, video_feat, pool_text_feat, text_mask, video_mask, pseudo_text, pool_pseudo_text, pseudo_text_mask

    def generate_gauss_weight(self, props_len, center, width):

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def _mask_feat(self, feat, feat_len, weights=None, mask_rate = 0.3):
        
        masked_vec = []
        for i, l in enumerate(feat_len):
            l = int(l)
            # num_masked_vec = max(l // 3, 1) 
            num_masked_vec = max(int(l * mask_rate), 1) 
            masked_vec.append(torch.zeros([feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().detach().numpy() if weights is not None else None
            # choices = np.random.choice(np.arange(l), num_masked_vec, replace=False)
            choices = np.random.choice(np.arange(l), num_masked_vec, replace=False, p=p)
            # choices = torch.topk(weights[i, :l], k=num_masked_vec)[1]
            masked_vec[-1][choices] = 1

        masked_vec = torch.stack(masked_vec, 0).unsqueeze(-1)
        # out_feat = feat.masked_fill(masked_vec == 1, float("-inf"))
        out_feat = feat.masked_fill(masked_vec == 1, 0)
        return out_feat, masked_vec
    def _mask_moment(self, video_feat, video_mask, center, width):
        video_len = video_mask.sum(1)

        star, end =  torch.clamp(center-width/2, min=0), torch.clamp(center+width/2, max=1)
        star, end = (video_len * star).to(torch.int), (video_len * end).to(torch.int)
        masked_vec = torch.zeros(video_mask.shape).byte().cuda()

        for i, l in enumerate(video_len):
            if star[i] < end[i]:
                masked_vec[i][star[i]:end[i]] = 1
            elif star[i] == end[i]:
                masked_vec[i][star[i]] = 1
            else:
                masked_vec[i][end[i]:star[i]] = 1
        masked_vec = masked_vec.unsqueeze(-1)
        video_feat = video_feat.masked_fill(masked_vec == 1, 0)
        return video_feat, masked_vec

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.contiguous()
        return text_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat


    def get_marginal_loss(self, cosx, m, s):
        sinx = torch.sqrt(1.0 - torch.pow(cosx, 2))
        cosm = math.cos(m)
        sinm = math.sin(m)
        return (cosx * cosm - sinx * sinm)/s
        
    def get_similarity_loss(self, text_feat, pool_text_feat, video_feat, text_mask, video_mask, shaped=False, video_attention_mask=None, v2w=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        t2v_logits, v2t_logits, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, dis_text_feat, dis_video_feat = self.get_similarity_logits(text_feat, pool_text_feat, video_feat, text_mask, video_mask, video_attention_mask=video_attention_mask, gauss=self.do_gauss, v2w=v2w)
        
        logit_scale = self.clip.logit_scale.exp()

        # t2v_logits = self.get_marginal_loss(t2v_logits, 0.25, 0.05)/logit_scale
        # v2t_logits = self.get_marginal_loss(v2t_logits, 0.25, 0.05)/logit_scale

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(v2t_logits * logit_scale)
        
        loss = (loss_t2v + loss_v2t) / 2

        return loss, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, dis_text_feat, dis_video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()