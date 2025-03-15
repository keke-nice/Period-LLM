# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
import numpy as np
import os
import json
from pathlib import Path
from llama.tokenizer import Tokenizer
from llama import Transformer, ModelArgs
import copy
from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
import clip
from torch.utils.hooks import unserializable_hook

np.set_printoptions(threshold=np.inf)
sys.path.append('..')

# 将 adjust_grad 函数移到全局作用域
# @unserializable_hook
# def adjust_grad(grad, h_Q, iter_num, max_iter, beta):
    
#     avg = h_Q.mean(dim=1, keepdim=True) 
#     c_bar = avg.mean()  # 全局平均值
#     adjustment = torch.ones_like(avg)
#     adjustment[avg < c_bar] *= (1 + beta * torch.exp(iter_num / max_iter))
#     adjustment[avg >= c_bar] *= 1
#     grad *= adjustment
    
#     return grad

@unserializable_hook
def adjust_grad(grad, h_Q, k):
    # 计算最后一个维度的平均值
    avg = h_Q.mean(dim=1, keepdim=True)  # [1, 1, 4096]
    adjustment = torch.ones_like(avg)
    
    # 增大平均值较小的通道的梯度，缩小平均值较大的通道的梯度
    adjustment[avg < avg.mean()] *= 1 + k
    adjustment[avg >= avg.mean()] *= 1

    grad *= adjustment
    
    return grad

# 创建一个类来封装钩子的逻辑
class AdjustGradHook:
    def __init__(self, h_Q, k):
        self.h_Q = h_Q
        self.k = k
    def __call__(self, grad):
        return adjust_grad(grad, self.h_Q, self.k)
    
    def update_k(self, new_k):
        self.k = new_k

class BaseNet(nn.Module):
    def __init__(self, args, my_type, input_type, device, image_size,
                 med_config='configs/med_config.json', 
                 visual='swin_b',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        super(BaseNet, self).__init__()
        self.image_size = image_size
        self.visual = visual
        self.vit_grad_ckpt = vit_grad_ckpt
        self.vit_ckpt_layer = vit_ckpt_layer
        self.half_train = False
        self.input_type = input_type
        self.device = device
        self.my_type = my_type
        self.k = 0
   

        self.init_visual_encoder()
       
        self.init_llama()
        self.inin_visual_project()
        # Tokenizer
        tokenizer_path = '../LLAMA/llama-2-7b/tokenizer.model'
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        # 用于存储钩子句柄的列表
        self.hook_handles = []


    def load_llama(self, llama_path: str):
        model_args: ModelArgs = ModelArgs(
            max_seq_len=512, max_batch_size=self.llama_max_batch_size, 
            w_bias=True, w_lora=False, lora_rank=16, 
            w_new_gate=False, vocab_size=32000, dim=4096, 
            n_heads=32, n_layers=32, multiple_of=256, norm_eps=1e-05
        )
        llama = Transformer(model_args)
        ckpts = sorted(Path(llama_path).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            llama.load_state_dict(ckpt, strict=False)
        return llama

    def init_llama(self, llama_path='../LLAMA/llama-2-7b'):

        self.LLM_Loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # 参数设置
        self.query_layer_index = [1, 5, 9, 13, 17, 21, 25]
        self.query_layer = 31
        self.max_words = 512
        self.v_embed_dim = 768
        self.v_depth = 8
        self.v_num_heads = 16
        self.v_mlp_ratio = 4.0
        self.num_cam = 6
        self.image_channel = 512
        self.position_channel = 0
        self.llama_max_batch_size = 2
        self.batch_size = 1
        # load llama
        self.llama = self.load_llama(llama_path)
        for name, para in self.llama.named_parameters():
            para.requires_grad = False
        for name, para in self.llama.named_parameters():
            if 'bias' in name:
                para.data = para.data.float()
                para.requires_grad = True
        if self.half_train:
            self.llama = self.llama.half()

    def init_visual_encoder(self, ckp_filename='./pretrained_models/tag2text_swin_14m.pth'):
        if self.visual == 'clip':
            self.vision_width = 512
            self.visual_encoder, preprocess = clip.load("ViT-B/32", device=self.device)
            self.fc_norm = nn.LayerNorm(self.vision_width)
        elif self.visual == 'swin_b':
            if self.image_size == 224:
                vision_config_path = 'configs/swin/config_swinB_224.json'
            elif self.image_size == 384:
                vision_config_path = 'configs/swin/config_swinB_384.json'
            with open(vision_config_path, 'r') as f:
                vision_config = json.load(f)
            assert self.image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            self.vision_width = vision_config['vision_width']
            window_size = vision_config['window_size']
            checkpoint = torch.load(ckp_filename, map_location='cpu')
            state_dict = checkpoint['model']
            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * window_size - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]
            keys = state_dict.keys()
            new_state_dict = {}
            for key in keys:
                if key.startswith('visual_encoder.'):
                  new_key = key[len('visual_encoder.'):]
                  new_state_dict[new_key] = state_dict[key]
            self.visual_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                    patch_size=4,
                                    in_chans=3,
                                    embed_dim=vision_config['embed_dim'],
                                    depths=vision_config['depths'],
                                    num_heads=vision_config['num_heads'],
                                    window_size=vision_config['window_size'],
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    drop_rate=0.0,
                                    drop_path_rate=0.1,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)
            model_state_dict = self.visual_encoder.state_dict()
            msg = self.visual_encoder.load_state_dict(new_state_dict,strict=False)
            print('load checkpoint from %s'%ckp_filename)  
            self.fc_norm = nn.LayerNorm(self.vision_width)

    def inin_visual_project(self):
        self.visual_proj = nn.Sequential(
            nn.Conv1d(self.vision_width, 4096, kernel_size=1, stride=1, padding=0, groups=64),
            nn.ReLU(),
            nn.BatchNorm1d(4096)
        )

    def tokenize(self, question, label, video_flag):
        tokened_input = []
        tokened_label = []
        tokened_infer = []
        # print(question)
        # print(len(question))
        for index in range(len(question)):
            input0 = question[index]
            if isinstance(input0, tuple):
                input0 = input0[0]
            # print(input0)
            input2 = label[index]
            if isinstance(input2, tuple):
                input2 = input2[0]
            input2 = input0 + input2
            input1 = torch.tensor(self.tokenizer.encode(input0, bos=True, eos=False), dtype=torch.int64) #15
            input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64) #27
            input3 = torch.tensor(self.tokenizer.encode(input0, bos=True, eos=True), dtype=torch.int64)
            if video_flag:
                input1 = torch.cat((torch.zeros(self.token_image_len, dtype=torch.int64) - 1, input1)) #16
                input2 = torch.cat((torch.zeros(self.token_image_len, dtype=torch.int64) - 1, input2)) #28
                input3 = torch.cat((torch.zeros(self.token_image_len, dtype=torch.int64) - 1, input3))
            padding = self.max_words - input2.shape[0]
            padding_infer =  self.max_words - input3.shape[0]
            input3 = torch.cat((input3, torch.zeros(padding_infer, dtype=torch.int64) - 1))
            if padding > 0:
                input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1)) #127
            elif padding < 0:
                input2 = input2[:self.max_words]
            labels = copy.deepcopy(input2)
            labels[:len(input1)] = -1
            input2_mask = input2.ge(0)
            label_mask = labels.ge(0)
            input3_mask = input3.ge(0)
            input2[~input2_mask] = 0
            labels[~label_mask] = 0
            input3[~input3_mask] = 0
            tokened_input.append(input2)
            tokened_label.append(labels)
            tokened_infer.append(input3)
        return tokened_input, tokened_label, tokened_infer

    def LLM_Train(self, img_feats, tokened_Q, tokened_GT, video_flag):
        # token_image_len = img_feats.size(0) #[2,512]
        ## 0. 视觉映射
        if video_flag:
            em_project_input = img_feats.view(self.batch_size * self.token_image_len, -1, 1)#[2,512,1]
            if self.visual=='clip':
                em_project_input = em_project_input.float()
            em_project_input = self.visual_proj(em_project_input)#[2,4096,1]
            em_project_input = em_project_input.view(self.batch_size, self.token_image_len, 4096)#[2,1,4096]
        else:
            self.token_image_len=0
            em_project_input = 0
        with torch.cuda.amp.autocast():
            ## 求token embedding
            h_Q = self.llama.tok_embeddings(tokened_Q) #[2,127,4096]
            _bsz, seqlen = tokened_Q.shape
            h_Q[:, :self.token_image_len] = h_Q[:, :self.token_image_len] + em_project_input
  
            freqs_cis = self.llama.freqs_cis.to(h_Q.device) #[1024,64]
            freqs_cis = freqs_cis[:seqlen]#[127,64]
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h_Q.device)#[1,1,127,127]
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h_Q)#[1,1,127,127]
            for layer in self.llama.layers:
                h_Q = layer(h_Q, 0, freqs_cis, mask)#[2,127,4096]
            h_Q = self.llama.norm(h_Q)
            # 在前向传播之后，反向传播之前注册钩子
            if not self.hook_handles:
                self.adjust_grad_hook = AdjustGradHook(h_Q, self.k)
                hook_handle = h_Q.register_hook(self.adjust_grad_hook)
                self.hook_handles.append(hook_handle)
            output = self.llama.output(h_Q)#[2,127,32000]
            output = output[:, :-1, :]
            labels = tokened_GT[:, 1:]
            
           
            if labels.sum() == 0:
                c_loss = output.mean() * 0
            else:
                assert self.llama.vocab_size == 32000
                c_loss = self.LLM_Loss(output.reshape(-1, self.llama.vocab_size), labels.flatten())    
   
            losses_QA = c_loss
        return losses_QA
    
    def update_k(self, new_k):
        self.k = new_k
        if hasattr(self, 'adjust_grad_hook'):
            self.adjust_grad_hook.update_k(new_k)

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) 
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p #
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    @torch.inference_mode()
    def forward_inference(self, img_feats, tokens, start_pos: int):
        # tokens
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens) #[1,30,4096]
        # img_feats
        # token_image_len = img_feats.shape[0]
        if self.input_type=='video':
            em_project_input = img_feats.view(self.batch_size * self.token_image_len, -1, 1)#[1,512,1]
            em_project_input = self.visual_proj(em_project_input) #[1,4096,1]
            em_project_input = em_project_input.view(self.batch_size, self.token_image_len, 4096)#[1,1,4096]
        elif self.input_type=='text':
            em_project_input=0
        # fusion
        if start_pos < self.token_image_len:
            h[:, :self.token_image_len] = h[:, :self.token_image_len] + em_project_input
        freqs_cis = self.llama.freqs_cis.to(self.device)
        freqs_cis = freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=self.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.llama.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])
        return output.float()

    @torch.inference_mode()
    def generate(
            self, img_feats, token_input,
            max_gen_len: int = 512,
            temperature: float = 0.1,
            top_p: float = 0.75,
    ):
        # token_image_len = len(img_feats)
        params = self.llama.params
        assert self.batch_size <= params.max_batch_size, (self.batch_size, params.max_batch_size)
        # assert len(img_feats) == len(token_input)
        prompts = [t[:t.gt(0).sum()+self.token_image_len-1] for t in token_input]
                # 找到token最大最小
        min_prompt_size = min([t.gt(0).sum()+self.token_image_len-1 for t in token_input])
        max_prompt_size = max([t.gt(0).sum()+self.token_image_len-1 for t in token_input])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)#256+30
        tokens = torch.full((self.batch_size, total_len), self.tokenizer.pad_id).to(self.device).long()
        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).to(self.device).long()
        input_text_mask = tokens != self.tokenizer.pad_id 
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(img_feats, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if self.batch_size == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)] 
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

    def Inference(self, x, question_all):
        question_all = list(question_all)
        if self.input_type=='video':
            video_flag = True
            # 视觉特征提取
            if self.visual == 'clip':
                em = self.visual_encoder.encode_image(x.squeeze(0))
            elif self.visual == 'swin_b':
                em = self.visual_encoder(x.squeeze(0))
                em = self.fc_norm(em[:, 1:, :].mean(dim=1)) #[token_image_len, 1024]
            self.token_image_len = em.shape[0] 
        elif self.input_type=='text':
            video_flag = False
            em=None
            self.token_image_len=0
        # Tokenizer
        _, _, token_input = self.tokenize(question_all, question_all, video_flag)
        token_input = torch.stack(token_input, dim=0).to(self.device)#[1,128]
        caption = self.generate(em, token_input)
        return caption


    def forward(self, x, question_all, caption_GT):
        question_all = list(question_all)
        caption_GT = list(caption_GT)

        if self.input_type=='video' or self.input_type=='double':
            video_flag = True
            # 视觉特征提取
            with torch.no_grad():
                if self.visual == 'clip':
                    em = self.visual_encoder.encode_image(x.squeeze(0))
                elif self.visual == 'swin_b':
                    em = self.visual_encoder(x.squeeze(0))
                    em = self.fc_norm(em[:, 1:, :].mean(dim=1)) #[token_image_len, 1024]
            self.token_image_len = em.shape[0] 
        elif self.input_type=='text': 
            video_flag = False
            em = None
        # Tokenizer
        token_input, token_label, _ = self.tokenize(question_all, caption_GT, video_flag)
        token_input = torch.stack(token_input, dim=0).to(device=self.device)#[2,127]
        token_label = torch.stack(token_label, dim=0).to(device=self.device)#[2,127]
        # LLM
        loss_QA = self.LLM_Train(em, token_input, token_label, video_flag)

        return loss_QA