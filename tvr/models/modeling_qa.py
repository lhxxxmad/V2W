from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cgitb import text

import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from .until_module import PreTrainedModel, AllGather, CrossEn, Slip
from .module_cross import CrossConfig, Transformer as TransformerClip
from .module_clip import CLIP, convert_weights
from .loss import CrossEn
from .transformer import DualTransformer
import torch.nn.functional as F
import pdb
import numpy as np
from .PDE import DisTrans
from .loss import compute_dis_contrast
logger = logging.getLogger(__name__)
allgather = AllGather.apply


class EMCL4QAPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(EMCL4QAPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16")
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        contain_frame_position = False
        for key in state_dict.keys():
            if key.find("frame_position_embeddings") > -1:
                contain_frame_position = True
                break
        if contain_frame_position is False:
            for key, val in clip_state_dict.items():
                if key == "positional_embedding":
                    state_dict["frame_position_embeddings.weight"] = val.clone()
                    continue
                if key.find("transformer.resblocks") == 0:
                    num_layer = int(key.split(".")[2])
                    # cut from beginning
                    if num_layer < task_config.cross_num_hidden_layers:
                        state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class EMCL4QA(EMCL4QAPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(EMCL4QA, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.dropout = nn.Dropout(0.1)

        self.emcl = Slip(k=task_config.K,
                         stage_num=task_config.stage_num,
                         momentum=task_config.momentum,
                         lamd=task_config.lamd,
                         beta=task_config.beta)

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b
                            in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        cross_config.max_position_embeddings = context_length
        self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
        self.transformerClip = TransformerClip(width=transformer_width,
                                                   layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )

        hidden_size = transformer_width * 8
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,
                      hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, task_config.num_labels)
        )

        self.v_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.t_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.loss_fct = CrossEn()
        self.sample_num = self.task_config.sample_num
        if vit:
            self.mean_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.mean_proj = nn.Linear(embed_dim, embed_dim)
        self.embd_mode = 'wti'
        self.sal_pred = 'sa+mlp'
        self.interact_mode = 'FGW'
        self.v_w = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )

        self.t_w = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )

        self.apply(self.init_weights)

        self.video_gau_trans = DisTrans(transformer_width, transformer_heads)
        self.text_gau_trans = DisTrans(transformer_width, transformer_heads)
        
        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.text_saliency_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, self.task_config.max_words))
        self.video_saliency_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, self.task_config.max_frames))
        self.temporal_order_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, self.task_config.max_frames))
        self.attn = nn.MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        self.rec_video_trans = DualTransformer()
        self.rec_text_trans = DualTransformer()
        self.temporal_trans = DualTransformer()
        self.mse_loss = nn.MSELoss()
    def forward(self, input_ids, token_type_ids, text_mask, video, video_mask=None,
                labels=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, n_v, d, h, w = video.shape
        video = video.view(b * n_v, d, h, w)
        video_frame = n_v

        text_feat, video_feat = self.get_sequence_video_feat(input_ids, token_type_ids, text_mask, video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            labels = allgather(labels, self.task_config)
            # class_logits, cls_loss = self.calc_loss(class_logits, labels)

            video_feat = allgather(video_feat, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            text_feat = allgather(text_feat, self.task_config)
            text_mask = allgather(text_mask, self.task_config)

            torch.distributed.barrier()
        retrieval_loss, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, _, __ = self.get_similarity_loss(text_feat, None, video_feat, text_mask, video_mask, shaped=True)
        # retrieve_logits = self.get_similarity_logits(text_feat, None, video_feat, text_mask, video_mask, text_weight, video_weight)

        txt_mu = self._mean_pooling_for_similarity_sequence(txt_mu, text_mask)
        txt_logsigma = self._mean_pooling_for_similarity_sequence(txt_logsigma, text_mask)
        vid_mu = self._mean_pooling_for_similarity_visual(vid_mu, video_mask)
        vid_logsigma = self._mean_pooling_for_similarity_visual(vid_logsigma, video_mask)

        dis_cl_loss = compute_dis_contrast(txt_mu, torch.exp(txt_logsigma), vid_mu, torch.exp(vid_logsigma))

        # rec_video_loss, rec_text_loss = self.get_rec_loss(text_feat, video_feat, text_mask, video_mask, text_weight, video_weight)
        # temporal_loss = self.get_temporal_order_loss(text_feat, video_feat, text_mask, video_mask, text_weight, video_weight)

        # loss = loss + (sim_loss + (rec_video_loss + rec_text_loss) /2.0 + temporal_loss) * 0.5
        class_logits = self.get_cl_logits(video_feat, video_mask, text_feat, text_mask, text_weight, video_weight)
        class_logits, cls_loss = self.calc_loss(class_logits, labels)
        # loss = cls_loss + (retrieval_loss + dis_cl_loss) *0.5

        final_loss = cls_loss + (retrieval_loss + dis_cl_loss) *0.5
        final_loss_dict = {'final_loss': final_loss.item(), 
                            'class_loss': retrieval_loss.item(),
                            'retrieval_loss': retrieval_loss.item(), 
                            'diss_loss': dis_cl_loss.item(), 
                            }
        if self.training:
            return final_loss, final_loss_dict
            # return loss
        else:
            return class_logits

    def get_similarity_logits(self, text_feat, cls, video_feat, text_mask, video_mask, video_attention_mask=None, gauss=False):
        video_mask = video_mask.squeeze()
        text_mask = text_mask.squeeze()
        # crossmodal_cyc_loss, inmodal_cyc_loss, inmodal_contras_loss = 0., 0., 0.
        text_weight, video_weight = None, None
        video_feat = video_feat.contiguous()

        # if video_attention_mask is not None:
        #     video_attention_mask = video_attention_mask.contiguous()
        #     if self.training:
        #         video_attention_mask = allgather(video_attention_mask, self.config)
        #     video_feat = video_feat * (video_attention_mask.unsqueeze(-1) + 1e-10)
            # video_feat = video_feat / video_attention_mask.sum(dim=-1, keepdim=True)
        if self.embd_mode == 'slip':
            v_weight = torch.einsum('ad,bvd->abv', [cls, video_feat])
            v_weight = torch.softmax(v_weight / self.config.temp, dim=-1)
            if video_attention_mask is None:
                v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])
            else:
                # v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask * video_attention_mask / video_attention_mask.sum(dim=-1, keepdim=True) ])
                v_weight = torch.einsum('abv,bv->abv', [v_weight, video_attention_mask])
                # v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])
            video_feat = torch.einsum('abv,bvd->abd', [v_weight, video_feat])
            a, d = cls.size()
            video_feat = video_feat.contiguous().view(-1, d)
            all_embedding = torch.cat((video_feat, cls), dim=0)
            all_embedding = self.slip(all_embedding, if_train=self.training)
            video_feat = all_embedding[:video_feat.size(0), :]
            text_feat = all_embedding[video_feat.size(0):, :]
            
            _t_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            _v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            
            _v_feat = _v_feat.view(a, -1, d)
            
            retrieve_logits = torch.einsum('ad,abd->ab', [_t_feat, _v_feat])
        elif self.embd_mode == 'xpool':
            video_feat = self.xpool(cls, video_feat, video_attention_mask=video_attention_mask)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            text_feat = cls / cls.norm(dim=-1, keepdim=True)            
            retrieve_logits = torch.bmm(text_feat.unsqueeze(1), video_feat.permute(1,2,0)).squeeze(1)

        elif self.embd_mode == 'cyc':
            bs = video_feat.size()[0]
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            video_feat = self.get_video_avg_feat(video_feat, video_mask)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

            text_feat = self.get_text_sep_feat(text_feat, text_mask).squeeze(1)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            retrieve_logits = torch.matmul(text_feat, video_feat.t())
            # if self.training:
            #     text_logits = torch.matmul(text_feat, text_feat.t()) * self.clip.logit_scale.exp()
            #     video_logits = torch.matmul(video_feat, video_feat.t()) * self.clip.logit_scale.exp()

            #     crossmodal_cyc_loss = (retrieve_logits* self.clip.logit_scale.exp() - retrieve_logits.T* self.clip.logit_scale.exp()).square().mean() / (self.clip.logit_scale.exp() * self.clip.logit_scale.exp())
            #     inmodal_cyc_loss = (video_logits - text_logits).square().mean() / (self.clip.logit_scale.exp() * self.clip.logit_scale.exp())

            #     inmodal_contras_loss = (self.loss_fct(text_logits) + self.loss_fct(video_logits) ) / 2 

        elif self.embd_mode == 'wti':

            video_mask = video_mask.squeeze()
            text_mask = text_mask.squeeze()

            ############################
            # SA
            # cross-attn
            # ################################################################
            B_t, N_t, D = text_feat.shape
            B_v, N_v, D = video_feat.shape

            # if B_t > B_v:
            #     pad_feat = torch.zeros(1, N_v, D).to(video_feat.device)
            #     pad_feat = pad_feat.repeat(B_t-B_v, 1, 1)
            #     video_feat = torch.cat([video_feat, pad_feat], dim=0)
            # if B_t < B_v:
            #     pad_feat = torch.zeros(1, N_t, D).to(text_feat.device)
            #     pad_feat = pad_feat.repeat(B_v-B_t, 1, 1)
            #     text_feat = torch.cat([text_feat, pad_feat], dim=0)

            if self.sal_pred == 'ca+mlp':
                cross_text_feat = self.attn(text_feat.permute(1,0,2), video_feat.permute(1,0,2), video_feat.permute(1,0,2))[0].permute(1,0,2)
                cross_video_feat = self.attn(video_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)
            elif self.sal_pred == 'trans':
                cross_text_feat = self.saliency_text_trans(video_feat.permute(1,0,2), text_feat.permute(1,0,2)).permute(1,0,2)
                # cross_text_feat = self.rec_text_trans1(text_feat, None, video_feat, None, decoding=1)[1]
                cross_video_feat = self.saliency_video_trans(text_feat.permute(1,0,2), video_feat.permute(1,0,2)).permute(1,0,2)
                # cross_video_feat = self.rec_video_trans1(video_feat, None, text_feat, None,  decoding=1)[1]
            elif self.sal_pred == 'mlp':
                cross_text_feat = text_feat
                cross_video_feat = video_feat
            elif self.sal_pred == 'sa+mlp':
                cross_text_feat = self.attn(text_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)
                cross_video_feat = self.attn(video_feat.permute(1,0,2), video_feat.permute(1,0,2), video_feat.permute(1,0,2))[0].permute(1,0,2)

            # if B_t < B_v:
            #     text_feat = text_feat[: B_t, ::]
            #     cross_text_feat = cross_text_feat[: B_t, ::]
            # if B_t > B_v:
            #     video_feat = video_feat[: B_v, ::]
            #     cross_video_feat = cross_video_feat[: B_v, ::]

            # saliency token
            if gauss:
                # text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
                # text_weight =  self.text_saliency_fc(cross_text_feat[:,-1])
                # video_weight =  self.video_saliency_fc(cross_video_feat[:,-1])
                props = self.moment_fc(cross_video_feat[:,-1])
                cross_video_feat = cross_video_feat[:, : -1]
                cross_text_feat = cross_text_feat[:, : -1]

                text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
                video_weight = self.video_weight_fc(cross_video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v

                text_feat = text_feat[:, : -1]
                video_feat = video_feat[:, : -1]
                text_mask = text_mask[:, : -1]
                video_mask = video_mask[:, : -1]
            else:
                # MLP
                # text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
                # Cross-Attn
                text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t

                # MLP
                # video_weight = self.video_weight_fc(video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v
                # Cross-Attn
                video_weight = self.video_weight_fc(cross_video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v
            
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
            # text_weight = torch.sigmoid(text_weight)  # B_t x N_t            

            video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
            video_weight = torch.softmax(video_weight, dim=-1)  # B_v x N_v
            # video_weight = torch.sigmoid(video_weight)  # B_v x N_v
            # ################################################################
            
            # probability distribution sampling
            B,N,C = text_feat.shape
            txt_mu, txt_logsigma, _ = self.text_gau_trans(text_feat, weight=text_weight)
            samples = [txt_mu]
            for _ in range(self.sample_num-1):
                eps = torch.randn(B, N, C, device=txt_mu.device)
                sample = txt_mu + torch.exp(txt_logsigma) * eps
                samples.append(sample)
            # pdb.set_trace()
            dis_text_feat = torch.cat(samples).view(B, self.sample_num, N, C).mean(dim=1)
            text_feat = text_feat + F.dropout(dis_text_feat, p=0.1)
            # text_feat = self.dis_fc1(text_feat)
            # text_mask = text_mask.unsqueeze(1).expand(B, self.sample_num, -1).reshape(B * self.sample_num, N)

            B,N,C = video_feat.shape
            vid_mu, vid_logsigma, _ = self.video_gau_trans(video_feat, weight=video_weight)
            samples = [vid_mu]
            for _ in range(self.sample_num-1):
                eps = torch.randn(B, N, C, device=vid_mu.device)
                sample = vid_mu + torch.exp(vid_logsigma) * eps
                samples.append(sample)
            dis_video_feat = torch.cat(samples).view(B, self.sample_num, N, C).mean(dim=1)
            video_feat = video_feat + F.dropout(dis_video_feat, p=0.1)
            # video_feat = self.dis_fc2(video_feat)
            # video_mask = video_mask.unsqueeze(1).expand(B, self.sample_num, -1).reshape(B * self.sample_num, N)

            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

            retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
            retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
            retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
            text_sum = text_mask.sum(-1)
            video_sum = video_mask.sum(-1)

            if self.interact_mode == 'FGW':
                # weighted token-wise interaction
                t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
                t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

                v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
                v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
                retrieve_logits = (t2v_logits + v2t_logits) / 2.0            
            elif self.interact_mode == 'FGM':
                t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
                v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv

                t2v_logits = torch.einsum('abt,at->abt', [t2v_logits, text_weight])
                v2t_logits = torch.einsum('abv,bv->abv', [v2t_logits, video_weight])
                
                t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
                v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
                retrieve_logits = (t2v_logits + v2t_logits) / 2.0

            elif self.interact_mode == 'CGW':
                text_feat = torch.einsum('atd,at->atd', [text_feat, text_mask])
                text_feat = torch.einsum('atd,at->ad', [text_feat, text_weight])
                video_feat = torch.einsum('bvd,bv->bvd', [video_feat, video_mask])
                video_feat = torch.einsum('bvd,bv->bd', [video_feat, video_weight])
                retrieve_logits = torch.einsum('ad,bd->ab', [text_feat, video_feat])

            elif self.interact_mode == 'CGM':
                text_feat = torch.einsum('atd,at->atd', [text_feat, text_mask])
                video_feat = torch.einsum('bvd,bv->bvd', [video_feat, video_mask])

                text_feat = torch.einsum('atd,at->atd', [text_feat, text_weight]) + text_feat
                video_feat = torch.einsum('bvd,bv->bvd', [video_feat, video_weight]) + video_feat

                text_feat = torch.sum(text_feat, dim=1) / (text_sum.unsqueeze(1))
                video_feat = torch.sum(video_feat, dim=1) / (video_sum.unsqueeze(1))
                retrieve_logits = torch.einsum('ad,bd->ab', [text_feat, video_feat])
                
            # retrieve_logits = self.get_marginal_loss(retrieve_logits, 0.25, 0.05)
        return retrieve_logits, retrieve_logits.T, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, text_feat, video_feat

    def get_similarity_loss(self, text_feat, cls, video_feat, text_mask, video_mask, shaped=False, video_attention_mask=None):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        t2v_logits, v2t_logits, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, text_feat, video_feat = self.get_similarity_logits(text_feat, cls, video_feat, text_mask, video_mask, video_attention_mask=video_attention_mask, gauss=False)
        
        logit_scale = self.clip.logit_scale.exp()

        # t2v_logits = self.get_marginal_loss(t2v_logits, 0.25, 0.05)/logit_scale
        # v2t_logits = self.get_marginal_loss(v2t_logits, 0.25, 0.05)/logit_scale

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(v2t_logits * logit_scale)
        
        loss = (loss_t2v + loss_v2t) / 2

        return loss, text_weight, video_weight, txt_mu, txt_logsigma, vid_mu, vid_logsigma, text_feat, video_feat

    def get_temporal_order_loss(self, text_feat, video_feat, text_mask, video_mask, text_weight, video_weight):
        B, T, D = video_feat.shape
        shuffle_idx = torch.from_numpy(np.random.permutation(np.arange(T))).to(video_feat.device)
        shuffle_video_feat = video_feat[:,shuffle_idx]
        shuffle_video_mask = video_mask[:,shuffle_idx]
        recover_idx = torch.argsort(shuffle_idx)
        # shuffle_out = self.temporal_trans(text_feat, None, shuffle_video_feat, None,  decoding=2, gauss_weight=text_weight)[1]
        shuffle_out = self.temporal_trans(text_feat, text_mask, shuffle_video_feat, shuffle_video_mask,  decoding=2, gauss_weight=text_weight)[1]
        temporal_order_pred = self.temporal_order_fc(shuffle_out).squeeze()
        shuffle_idx = shuffle_idx.unsqueeze(0).repeat(B,1)
        # temporal_loss = self.BCE_loss(temporal_order_pred, shuffle_idx, shuffle_video_mask)
        # temporal_loss = nn.BCEWithLogitsLoss(pos_weight=shuffle_video_mask)(temporal_order_pred, shuffle_idx.float())
        # temporal_loss = nn.CrossEntropyLoss()(temporal_order_pred, shuffle_idx)
        temporal_loss = self.nll_loss(temporal_order_pred, shuffle_idx, shuffle_video_mask, weights=video_weight).mean()
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
        if weights is None:
            nll_loss = nll_loss.masked_fill(mask == 0, 0)
            nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
        else:
            nll_loss = (nll_loss * weights).sum(dim=-1)

        return nll_loss

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
            choices = np.random.choice(np.arange(l), num_masked_vec, replace=False, p=p)
            masked_vec[-1][choices] = 1

        masked_vec = torch.stack(masked_vec, 0).unsqueeze(-1)
        # out_feat = feat.masked_fill(masked_vec == 1, float("-inf"))
        out_feat = feat.masked_fill(masked_vec == 1, 0)
        return out_feat

    def get_rec_loss(self, text_feat, video_feat, text_mask, video_mask, text_weight, video_weight):
        # text_weight = allgather(text_weight, self.config)
        # video_weight = allgather(video_weight, self.config)

        # random mask_rete 
        # text_mask_rate = np.random.uniform(0, 1.0)
        # video_mask_rate = np.random.uniform(0, 1.0)
        # masked_video = self._mask_feat(video_feat, video_mask.sum(1), video_weight, mask_rate=video_mask_rate )
        # masked_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=text_mask_rate)

        # mask_rete 
        masked_video = self._mask_feat(video_feat, video_mask.sum(1), video_weight, mask_rate=0.7)
        masked_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=0.7)

        # #  p = random
        # masked_video = self._mask_feat(video_feat, video_mask.sum(1), mask_rate=self.config.video_mask_rate)
        # masked_text = self._mask_feat(text_feat, text_mask.sum(1), mask_rate=self.config.text_mask_rate)
        # w/ mask
        # rec_video = self.rec_trans(masked_video, video_mask, text_feat, text_mask, decoding=1, gauss_weight=text_weight)[1]
        # rec_text = self.rec_trans(video_feat, video_mask, masked_text, text_mask, decoding=2, gauss_weight=video_weight)[1]

        rec_video = self.rec_video_trans(text_feat, None, masked_video, None,  decoding=2, gauss_weight=text_weight)[1]
        rec_text = self.rec_text_trans(video_feat, None, masked_text, None, decoding=2, gauss_weight=video_weight)[1]

        # w/o gauss weight
        # rec_video = self.rec_video_trans(text_feat, None, masked_video, None,  decoding=2, gauss_weight=text_weight)[1]
        # rec_text = self.rec_text_trans(video_feat, None, masked_text, None, decoding=2, gauss_weight=video_weight)[1]

        rec_video_loss = self.mse_loss(rec_video, video_feat)
        rec_text_loss = self.mse_loss(rec_text, text_feat)
        return rec_video_loss, rec_text_loss


    def get_cl_logits(self, video_feat, video_mask, text_feat, text_mask, text_weight, video_weight):
        # if self.training:
        #     video_feat = allgather(video_feat, self.task_config)
        #     video_mask = allgather(video_mask, self.task_config)
        #     text_feat = allgather(text_feat, self.task_config)
        #     text_mask = allgather(text_mask, self.task_config)
        #     torch.distributed.barrier()

        # video_feat = self._mean_pooling_for_similarity_visual(video_feat, video_mask)
        # text_feat = self._mean_pooling_for_similarity_sequence(text_feat, text_mask)
        # video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        # video_feat = video_feat * video_mask_un
        # text_mask_un = text_mask.to(dtype=torch.float).unsqueeze(-1)
        # text_feat = text_feat * text_mask_un
        text_feat = torch.einsum(" atd,at->ad ", [text_feat, text_weight])
        video_feat = torch.einsum(" atd,at->ad ", [video_feat, video_weight])

        video_feat = self.v_proj(video_feat)
        text_feat = self.t_proj(text_feat)
        input = torch.cat((video_feat, text_feat), dim=1)
        pooled_output = self.dropout(input)
        logits = self.classifier(pooled_output)

        return logits
        
    def calc_loss(self, logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                        logits.view(-1, self.task_config.num_labels),
                        labels.view(-1))
        else:
            loss = 0
        return logits, loss

    def get_text_feat(self, input_ids, token_type_ids, text_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = input_ids.size(0)
        text_feat = self.clip.encode_text(input_ids, return_hidden=True, mask=text_mask)[1].float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        return text_feat

    def get_video_feat(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.view(bs_pair, -1, video_feat.size(-1))
        
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
        # video_feat = self.mean_proj(video_feat)
        video_feat = video_feat + video_feat_original

        return video_feat

    def get_sequence_video_feat(self, input_ids, token_type_ids, text_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        text_feat = self.get_text_feat(input_ids, token_type_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True, video_frame=video_frame)

        text_feat, video_feat = text_feat.contiguous(), video_feat.contiguous()

        # # cross_text_feat = self.attn(text_feat.permute(1,0,2), video_feat.permute(1,0,2), video_feat.permute(1,0,2))[0].permute(1,0,2)
        # # cross_video_feat = self.attn(video_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)
        # text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
        # # text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
        # # text_weight = self.text_saliency_fc(cross_text_feat[:,-1]).squeeze()  # B_t x 1 x D -> B_t x N_t

        # text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
        # text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
        # # text_weight = torch.sigmoid(text_weight)  # B_t x N_t            

        # video_weight = self.video_weight_fc(video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v
        # # video_weight = self.video_weight_fc(cross_video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v
        # # video_weight = self.video_saliency_fc(cross_video_feat[:,-1]).squeeze() # B_v x 1 x D -> B_v x N_v
        # video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
        # video_weight = torch.softmax(video_weight, dim=-1)  # B_v x N_v

        # # text_weight = torch.softmax(self.t_w(text_feat).squeeze(2), dim=-1)  # BxN_t
        # # video_weight = torch.softmax(self.v_w(video_feat).squeeze(2), dim=-1)  # BxN_t

        return text_feat, video_feat


    def _mean_pooling_for_similarity_visual(self, video_feat, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_out
    def _mean_pooling_for_similarity_sequence(self, text_feat, text_mask):
        text_mask_un = text_mask.to(dtype=torch.float).unsqueeze(-1)
        text_mask_un[:, 0, :] = 0.
        text_feat = text_feat * text_mask_un
        text_out = torch.sum(text_feat, dim=1) / torch.sum(text_mask_un, dim=1, dtype=torch.float)
        return text_out