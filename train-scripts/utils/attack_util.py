import torch
from sympy.abc import epsilon
import sys
from .util import *
import wandb
import torch.nn.functional as F

import numpy as np
import torch.nn as nn


def init_adv(k, tokenizer, all_embeddings, attack_type, device, batch=1, attack_init_embd=None):
    # Different attack types have different initializations (Attack types: add, insert)
    adv_embedding = torch.nn.Parameter(torch.randn([batch, k, 768])).to(device)

    if attack_init_embd is not None:
        # Use the provided initial adversarial embedding
        adv_embedding.data = attack_init_embd[:, 1:1 + k].data
    else:
        # Random sample k words from the vocabulary as the initial adversarial words
        # rand_init=torch.randn([batch, k, 768]).to(device)
        tmp_ids = torch.randint(0, len(tokenizer), (batch, k)).to(device)
        # print(tmp_ids)
        # tmp_ids=Inverter.embeddings_to_ids(rand_init)
        tmp_embeddings = id2embedding(tokenizer, all_embeddings, tmp_ids, device)
        tmp_embeddings = tmp_embeddings.reshape(batch, k, 768)
        adv_embedding.data = tmp_embeddings.data
    adv_embedding = adv_embedding.detach().requires_grad_(True)

    return adv_embedding


def soft_prompt_attack_batch(global_step, batch, words, model, model_orig, tokenizer, text_encoder, sampler, emb_0,
                             emb_p, start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k,
                             all_embeddings, attack_round, attack_type, attack_embd_type, attack_step, attack_lr,
                             attack_init=None, attack_init_embd=None):
    # print(f'======== Attack Round {attack_round} ========')
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                             x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                             start_code=code, n_samples=batch, till_T=t, verbose=False)

    if attack_init == 'latest':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], batch, attack_init_embd)
    elif attack_init == 'random':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], batch)

    attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)

    # Word Tokenization
    text_input = tokenizer(
        words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
    )

    # Construct input_ids and input_embeds for the ESD model
    idx = 0
    id_embd_list = []
    for word in words:
        orig_prompt_len = len(word.split())

        id_embd_dict = {}

        sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids[idx].unsqueeze(0).to(devices[0]), k,
                                                      orig_prompt_len)
        id_embd_dict['sot_id'] = sot_id
        id_embd_dict['mid_id'] = mid_id
        id_embd_dict['replace_id'] = replace_id
        id_embd_dict['eot_id'] = eot_id

        # Word embedding for the prompt
        text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids[idx].unsqueeze(0).to(devices[0]),
                                       devices[0])
        sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)
        id_embd_dict['sot_embd'] = sot_embd
        id_embd_dict['mid_embd'] = mid_embd
        id_embd_dict['eot_embd'] = eot_embd

        id_embd_list.append(id_embd_dict)
        idx += 1

    if attack_embd_type == 'condition_embd':
        raise ValueError('Batch attack does not support condition_embd')
        input_adv_condition_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
        adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)

    for i in range(attack_step):
        # ===== Randomly sample a time step from 0 to 1000 =====
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc) / ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((batch, 4, 64, 64)).to(devices[0])  # random inital noise

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
                                    int(t_enc))  # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                         emb_0.to(devices[0]))  # [batch, 4, 64, 64]
            e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                         emb_p.to(devices[0]))  # [batch, 4, 64, 64]
        # breakpoint()

        # Construct input_ids and input_embeds for the ESD model
        if attack_embd_type == 'word_embd':
            adv_id_embd_list = []
            for j in range(batch):
                adv_id_embd_dict = {}
                input_adv_word_embedding = construct_embd(k, adv_embedding[j, :, :].unsqueeze(0), attack_type,
                                                          id_embd_list[j]['sot_embd'], id_embd_list[j]['mid_embd'],
                                                          id_embd_list[j]['eot_embd'])
                adv_input_id = construct_id(k, id_embd_list[j]['replace_id'], attack_type, id_embd_list[j]['sot_id'],
                                            id_embd_list[j]['eot_id'], id_embd_list[j]['mid_id'])
                adv_id_embd_dict['input_adv_word_embedding'] = input_adv_word_embedding
                adv_id_embd_dict['adv_input_id'] = adv_input_id
                adv_id_embd_list.append(adv_id_embd_dict)

            # combine the adversarial word embedding and input_ids for the batch
            input_adv_word_embeddings = torch.cat(
                [adv_id_embd['input_adv_word_embedding'] for adv_id_embd in adv_id_embd_list], dim=0)
            adv_input_ids = torch.cat([adv_id_embd['adv_input_id'] for adv_id_embd in adv_id_embd_list], dim=0)
            input_adv_condition_embedding = \
                text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embeddings)[0]

        # get conditional score from ESD model with adversarial condition embedding
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                input_adv_condition_embedding.to(devices[0]))
        e_0.requires_grad = False
        e_p.requires_grad = False

        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[0]), e_p.to(devices[0]))
        loss.backward()
        attack_opt.step()

        wandb.log({'Attack_Loss': loss.item()}, step=global_step + i)
        wandb.log({'Train_Loss': 0.0}, step=global_step + i)

    if attack_embd_type == 'condition_embd':
        return input_adv_condition_embedding, adv_input_ids
    elif attack_embd_type == 'word_embd':
        return input_adv_word_embeddings, adv_input_ids
    else:
        raise ValueError('attack_embd_type must be either condition_embd or word_embd')


def soft_prompt_attack(global_step, word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p,
                       start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,
                       attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None,
                       attack_init_embd=None, attack_method='pgd'):
    '''
    Perform soft prompt attack on the ESD model
    Args:
        attack_type: str
            The type of attack (add or insert)
        attack_embd_type: str
            The type of adversarial embedding (condition_embd or word_embd)
        attack_step: int
            The number of steps for the attack
        attack_lr: float
            The learning rate for the attack
        attack_init: str
            The initialization method for the attack (latest or random)
        attack_init_embd: torch.Tensor
            The initial adversarial embedding
    '''
    try:
        orig_prompt_len = len(word.split())
    except:
        orig_prompt_len = len(word)
    if attack_type == 'add':
        k = orig_prompt_len

    # print(f'======== Attack Round {attack_round} ========')
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                             x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                             start_code=code, till_T=t, verbose=False)

    # Word Tokenization
    text_input = tokenizer(
        word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
    )
    sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k, orig_prompt_len)

    # Word embedding for the prompt
    text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
    sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)

    if attack_init == 'latest':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1, attack_init_embd)
    elif attack_init == 'random':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1)

    attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)

    if attack_embd_type == 'condition_embd':
        input_adv_condition_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
        adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)

    print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
    for i in range(attack_step):
        # ===== Randomly sample a time step from 0 to 1000 =====
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc) / ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # random inital noise

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
                                    int(t_enc))  # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z

            e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        # breakpoint()

        # Construct input_ids and input_embeds for the ESD model
        if attack_embd_type == 'word_embd':
            input_adv_word_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
            adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
            input_adv_condition_embedding = \
                text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]

        # get conditional score from ESD model with adversarial condition embedding
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                input_adv_condition_embedding.to(devices[0]))

        e_p.requires_grad = False

        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[0]), e_p.to(devices[0]))
        loss.backward()

        if attack_method == 'pgd':
            attack_opt.step()
        elif attack_method == 'fast_at':
            adv_embedding.grad.sign_()
            attack_opt.step()
        else:
            raise ValueError('attack_method must be either pgd or fast_at')

    if attack_embd_type == 'condition_embd':
        return input_adv_condition_embedding, adv_input_ids
    elif attack_embd_type == 'word_embd':
        return input_adv_word_embedding, adv_input_ids
    else:
        raise ValueError('attack_embd_type must be either condition_embd or word_embd')


def project(adv_embedding, all_embeddings):
    with torch.no_grad():
        # adv_embeddings = F.normalize(adv_embedding, p=2, dim=-1)
        # all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=-1)
        sim = F.cosine_similarity(adv_embedding.unsqueeze(2), all_embeddings, dim=-1)
        most_similar_idx = sim.argmax(dim=-1)
        proj_embeds = all_embeddings[0][most_similar_idx[0]].unsqueeze(0)
        return proj_embeds, most_similar_idx


def embedding_to_input_id(query_embedding, embedding_matrix):
    # Compute cosine similarity between query and all embeddings
    # print(embedding_matrix.size())
    with torch.no_grad():
        query_embedding = F.normalize(query_embedding.unsqueeze(2).detach().cpu(), p=2, dim=-1)
        embedding_matrix = F.normalize(embedding_matrix.detach().cpu(), p=2, dim=-1)
        similarities = query_embedding @ embedding_matrix.t()
        closest_idx = similarities.argmax(dim=-1)
    return closest_idx.squeeze(-1)


class EmbeddingInverter:
    def __init__(self, embedding_matrix, chunk_size=1024, fp16_mode=True):
        """
        支持三维输入的嵌入逆转换器
        :param embedding_matrix: 预训练词嵌入矩阵 [vocab_size, hidden_size]
        :param chunk_size: 分块处理大小（根据GPU显存调整）
        :param fp16_mode: 启用混合精度模式
        """
        # 初始化设置
        self.vocab_size, self.hidden_size = embedding_matrix.shape
        self.chunk_size = chunk_size
        self.fp16_mode = fp16_mode

        # CPU预处理（确保正确使用pin_memory）
        cpu_matrix = embedding_matrix.cpu().float()
        self.norm_matrix = F.normalize(cpu_matrix, p=2, dim=1)
        # self.norm_matrix =cpu_matrix

        # 混合精度处理
        if fp16_mode:
            self.norm_matrix = self.norm_matrix.half()

        # 固定内存加速数据传输
        self.norm_matrix = self.norm_matrix.contiguous().pin_memory()

    @torch.no_grad()
    def embeddings_to_ids(self, embeddings):
        """
        处理三维输入 [batch_size, text_length, hidden_size]
        :return: input_ids [batch_size, text_length]
        """
        device = embeddings.device
        batch_size, seq_len, _ = embeddings.shape

        # 展平处理以优化显存使用
        flattened_emb = embeddings.view(-1, self.hidden_size)  # [batch*seq_len, hidden]

        # 自动类型匹配
        input_dtype = self.norm_matrix.dtype
        normalized_emb = F.normalize(flattened_emb.to(input_dtype), p=2, dim=1)
        # normalized_emb=flattened_emb.to(input_dtype)

        # 结果存储
        best_ids = torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)
        best_scores = torch.full((batch_size * seq_len,), -float('inf'), device=device)

        # 分块处理词表

        for chunk_start in range(0, self.vocab_size, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.vocab_size)

            # 异步数据传输
            chunk = self.norm_matrix[chunk_start:chunk_end]

            chunk = chunk.to(device, non_blocking=True)
            # print(chunk.size())
            # 分块矩阵乘法

            # print(normalized_emb.size())
            # chunk_sim=torch.cosine_similarity(normalized_emb_new, chunk, dim=-1)
            chunk_sim = torch.matmul(normalized_emb, chunk.t())  # [batch*seq_len, chunk_size]

            # 增量更新最佳匹配
            current_max, current_ids = chunk_sim.max(dim=-1)
            update_mask = current_max > best_scores

            best_ids[update_mask] = current_ids[update_mask] + chunk_start
            best_scores[update_mask] = current_max[update_mask].to(torch.float32)

            # 显存清理
            del chunk, chunk_sim, current_max, current_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 恢复原始形状
        return best_ids.view(batch_size, seq_len)


# def soft_prompt_attack_max(global_step, word,retain_word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p,
#                            start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,
#                            attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None,
#                            attack_init_embd=None, attack_method='pgd'):
#     '''
#     Perform soft prompt attack on the ESD model
#     Args:
#         attack_type: str
#             The type of attack (add or insert)
#         attack_embd_type: str
#             The type of adversarial embedding (condition_embd or word_embd)
#         attack_step: int
#             The number of steps for the attack
#         attack_lr: float
#             The learning rate for the attack
#         attack_init: str
#             The initialization method for the attack (latest or random)
#         attack_init_embd: torch.Tensor
#             The initial adversarial embedding
#     '''
#     try:
#         orig_prompt_len = len(word.split())
#     except:
#         orig_prompt_len = len(word)
#     if attack_type == 'add':
#         k = orig_prompt_len
#
#     # print(f'======== Attack Round {attack_round} ========')
#     quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
#                                                              x, image_size, image_size, ddim_steps, s, ddim_eta,
#                                                              start_code=code, till_T=t, verbose=False)
#
#
#
#     # Word Tokenization
#     text_input = tokenizer(
#         word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
#     )
#     sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k, orig_prompt_len)
#
#
#     # Word embedding for the prompt
#     text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
#     sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)
#
#     if attack_init == 'latest':
#         adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1, attack_init_embd)
#     elif attack_init == 'random':
#         adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1)
#
#     attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)
#
#     if attack_embd_type == 'condition_embd':
#         input_adv_condition_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
#         adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#
#     print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
#     for i in range(attack_step):
#         # ===== Randomly sample a time step from 0 to 1000 =====
#         t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
#         og_num = round((int(t_enc) / ddim_steps) * 1000)
#         og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
#         t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
#         start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # random inital noise
#
#         with torch.no_grad():
#             # generate an image with the concept from ESD model
#             z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
#                                     int(t_enc))  # emb_p seems to work better instead of emb_0
#             # get conditional and unconditional scores from frozen model at time step t and image z
#             e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
#             e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
#         # breakpoint()
#
#         # Construct input_ids and input_embeds for the ESD model
#         if attack_embd_type == 'word_embd':
#             input_adv_word_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
#             pred_ids=embedding_to_input_id(input_adv_word_embedding,all_embeddings)
#             print(pred_ids)
#             adv_input_ids_0 = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#             adv_input_ids=pred_ids.to(devices[0])
#             print(adv_input_ids_0)
#             input_adv_condition_embedding = \
#             text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]
#
#         # get conditional score from ESD model with adversarial condition embedding
#         e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
#                                 input_adv_condition_embedding.to(devices[0]))
#         e_0.requires_grad = False
#         e_p.requires_grad = False
#
#         # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
#         loss = -criteria(e_n.to(devices[0]), e_p.to(devices[0]))
#         loss.backward()
#
#         if attack_method == 'pgd':
#             attack_opt.step()
#         elif attack_method == 'fast_at':
#             adv_embedding.grad.sign_()
#             attack_opt.step()
#         else:
#             raise ValueError('attack_method must be either pgd or fast_at')
#
#
#     if attack_embd_type == 'condition_embd':
#         return input_adv_condition_embedding, adv_input_ids
#     elif attack_embd_type == 'word_embd':
#         return input_adv_word_embedding, adv_input_ids
#     else:
#         raise ValueError('attack_embd_type must be either condition_embd or word_embd')
# def soft_prompt_attack_max(global_step, word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p,
#                            start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,
#                            attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None,
#                            attack_init_embd=None, attack_method='pgd'):
#     epsilon=8/255
#     '''
#     Perform soft prompt attack on the ESD model
#     Args:
#         attack_type: str
#             The type of attack (add or insert)
#         attack_embd_type: str
#             The type of adversarial embedding (condition_embd or word_embd)
#         attack_step: int
#             The number of steps for the attack
#         attack_lr: float
#             The learning rate for the attack
#         attack_init: str
#             The initialization method for the attack (latest or random)
#         attack_init_embd: torch.Tensor
#             The initial adversarial embedding
#     '''
#     try:
#         orig_prompt_len = len(word.split())
#     except:
#         orig_prompt_len = len(word)
#     if attack_type == 'add':
#         k = orig_prompt_len
#
#     # print(f'======== Attack Round {attack_round} ========')
#     quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
#                                                              x, image_size, image_size, ddim_steps, s, ddim_eta,
#                                                              start_code=code, till_T=t, verbose=False)
#
#     # Word Tokenization
#     text_input = tokenizer(
#         word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
#     )
#     k_max = 74
#     sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k_max, orig_prompt_len)
#
#     # Word embedding for the prompt
#
#     text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
#     # print(text_embeddings.shape)
#     sot_embd, mid_embd, main_embd, eot_embd = split_embd(text_embeddings, k_max, orig_prompt_len)
#     embd_edited=torch.cat((mid_embd, main_embd), dim=1)
#     if attack_init == 'latest':
#         adv_embedding = init_adv(k_max + 1, tokenizer, all_embeddings, attack_type, devices[0], 1, attack_init_embd)
#     elif attack_init == 'random':
#         adv_embedding = init_adv(k_max + 1, tokenizer, all_embeddings, attack_type, devices[0], 1)
#
#     # attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)
#
#     if attack_embd_type == 'condition_embd':
#         input_adv_condition_embedding = construct_embd_max(k, adv_embedding, main_embd, sot_embd, mid_embd, eot_embd)
#         adv_input_ids = construct_id_max(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#
#     print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
#     for i in range(attack_step):
#         # ===== Randomly sample a time step from 0 to 1000 =====
#         t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
#         og_num = round((int(t_enc) / ddim_steps) * 1000)
#         og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
#         t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
#         start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # random inital noise
#
#         with torch.no_grad():
#             # generate an image with the concept from ESD model
#             z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
#                                     int(t_enc))  # emb_p seems to work better instead of emb_0
#             # get conditional and unconditional scores from frozen model at time step t and image z
#             e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
#             e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
#         # breakpoint()
#
#         # Construct input_ids and input_embeds for the ESD model
#         if attack_embd_type == 'word_embd':
#             input_adv_word_embedding = construct_embd_max(k_max, adv_embedding, main_embd, sot_embd, mid_embd, eot_embd)
#             adv_input_ids = construct_id_max(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#             input_adv_condition_embedding = \
#                 text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]
#
#         # get conditional score from ESD model with adversarial condition embedding
#         e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
#                                 input_adv_condition_embedding.to(devices[0]))
#         e_0.requires_grad = False
#         e_p.requires_grad = False
#
#         # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
#         # loss = -criteria(e_n.to(devices[0]), e_p.to(devices[0]))
#         loss=F.kl_div(F.log_softmax(e_n.to(devices[0]),dim=1),F.softmax(e_p.to(devices[0]),dim=1),reduction='sum')
#         # loss.backward()
#
#         # if attack_method == 'pgd':
#         #     attack_opt.step()
#         # elif attack_method == 'fast_at':
#         #     adv_embedding.grad.sign_()
#         #     attack_opt.step()
#         # else:
#         #     raise ValueError('attack_method must be either pgd or fast_at')
#
#         grad_na = torch.autograd.grad(loss, [adv_embedding])[0]
#         if attack_method == 'pgd':
#             adv_embedding=adv_embedding.detach()+attack_lr*grad_na.detach()
#         elif attack_method == 'fast_at':
#             adv_embedding = adv_embedding.detach() + attack_lr * torch.sign(grad_na.detach())
#         input_adv_word_embedding = construct_embd_max(k_max, adv_embedding, main_embd, sot_embd, mid_embd, eot_embd)
#         input_adv_word_embedding=torch.min(torch.max(input_adv_word_embedding, text_embeddings - epsilon), text_embeddings + epsilon)
#         _, mid_embd_adv, main_embd_adv, _ = split_embd(input_adv_word_embedding, k_max, orig_prompt_len)
#         adv_embedding=torch.cat([mid_embd_adv, main_embd_adv], dim=1)-embd_edited
#
# #     return input_adv_word_embedding,adv_input_ids
def soft_prompt_attack_max(global_step, word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p,
                           start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,
                           attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None,
                           attack_init_embd=None, attack_init_embd_min=None, attack_method='pgd'):
    '''
       Perform soft prompt attack on the ESD model
       Args:
           attack_type: str
               The type of attack (add or insert)
           attack_embd_type: str
               The type of adversarial embedding (condition_embd or word_embd)
           attack_step: int
               The number of steps for the attack
           attack_lr: float
               The learning rate for the attack
           attack_init: str
               The initialization method for the attack (latest or random)
           attack_init_embd: torch.Tensor
               The initial adversarial embedding
       '''
    try:
        orig_prompt_len = len(word.split())
    except:
        orig_prompt_len = len(word)
    if attack_type == 'add':
        k = orig_prompt_len

    # print(f'======== Attack Round {attack_round} ========')
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                             x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                             start_code=code, till_T=t, verbose=False)

    # Word Tokenization
    text_input = tokenizer(
        word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
    )
    k_min = 1
    text_input_data = tokenizer(
        word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
    )
    sot_min_id, mid_min_id, replace_min_id, eot_min_id = split_id(text_input_data.input_ids.to(devices[0]), k_min,
                                                                  orig_prompt_len)

    sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k, orig_prompt_len)

    # Word embedding for the prompt
    text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
    sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)

    text_embeddings_data = id2embedding(tokenizer, all_embeddings, text_input_data.input_ids.to(devices[0]), devices[0])
    sot_min_embd, mid_min_embd, _, eot_min_embd = split_embd(text_embeddings_data, k_min, orig_prompt_len)
    # print(sot_min_embd.size())
    # print(mid_min_embd.size())
    # print(eot_min_embd.size())
    # attack_opt_min = torch.optim.Adam([adv_embedding_min], lr=attack_lr)

    if attack_init == 'latest':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1, attack_init_embd)
        adv_embedding_min = init_adv(k_min, tokenizer, all_embeddings, attack_type, devices[0], 1, attack_init_embd_min)
    elif attack_init == 'random':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1)
        adv_embedding_min = init_adv(k_min, tokenizer, all_embeddings, attack_type, devices[0], 1)

    attack_opt = torch.optim.Adam([adv_embedding_min], lr=attack_lr)

    print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
    for i in range(attack_step):
        # ===== Randomly sample a time step from 0 to 1000 =====
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc) / ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # random inital noise

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
                                    int(t_enc))  # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z

            e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        # breakpoint()

        # Construct input_ids and input_embeds for the ESD model
        if attack_embd_type == 'word_embd':
            input_adv_word_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
            adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
            input_adv_condition_embedding = \
                text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]

            input_adv_word_embedding_min = construct_embd(k_min, adv_embedding_min, attack_type, sot_min_embd,
                                                          mid_min_embd,
                                                          eot_min_embd)
            # print(input_adv_word_embedding_min.size())
            # print(adv_embedding_min.size())
            adv_input_min_ids = construct_id(k_min, replace_min_id, attack_type, sot_min_id, eot_min_id, mid_min_id)

            input_adv_condition_embedding_min = \
                text_encoder(input_ids=adv_input_min_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding_min)[0]

        e_adv = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                       input_adv_condition_embedding.to(devices[0]))
        e_min = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                  input_adv_condition_embedding_min.to(devices[0]))
        # assert torch.all(sampler.ddim_alphas[:-1] >= sampler.ddim_alphas[1:])
        # alpha_bar_t = sampler.ddim_alphas[int(t_enc)].to(devices[0])
        # e_p_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_p) / torch.sqrt(alpha_bar_t)
        # e_n_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_n) / torch.sqrt(alpha_bar_t)
        #
        # e_adv_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_adv) / torch.sqrt(alpha_bar_t)
        # e_min_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_min) / torch.sqrt(alpha_bar_t)

        # get conditional score from ESD model with adversarial condition embedding

        e_p.requires_grad = False

        # attack_opt.zero_grad()
        loss_min = criteria(e_min.to(devices[0]), e_p.to(devices[0]))
        # loss_min.backward()
        # adv_embedding_min.grad.sign_()
        # attack_opt.step()
        grad_min = torch.autograd.grad(loss_min, [adv_embedding_min])[0]
        adv_embedding_min.data = adv_embedding_min.data- attack_lr* torch.sign(grad_min.detach())

        # input_adv_word_embedding_min_t = construct_embd(k_min, adv_embedding_min, attack_type, sot_min_embd, mid_min_embd,
        #                                               eot_min_embd)
        # #
        # input_adv_condition_embedding_min_t = \
        #     text_encoder(input_ids=adv_input_min_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding_min_t)[0]
        # # # #
        # e_min = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
        #                           input_adv_condition_embedding_min_t.to(devices[0]))
        # e_min_pred_t = (z - torch.sqrt(1 - alpha_bar_t) * e_min) / torch.sqrt(alpha_bar_t)

        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        # criteria(e_adv.to(devices[0]), e_p.to(devices[0]).detach()) +
        loss = (criteria(e_adv.to(devices[0]), e_p.to(devices[0]).detach()) +criteria(
            e_adv.to(devices[0]), e_n.to(devices[0]).detach()) )
        grad = torch.autograd.grad(loss, [adv_embedding])[0]
        adv_embedding.data = adv_embedding.data - attack_lr * torch.sign(grad.detach())

        # (-loss).backward()
        # adv_embedding.grad.sign_()
        # attack_opt.step()

    # replace_id = embedding_to_input_id(adv_embedding, all_embeddings.squeeze(0)).to(sot_id.device)
    # adv_embedding.data = id2embedding(tokenizer, all_embeddings, replace_id.to(devices[0]),
    #                                       devices[0])

    input_adv_word_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd,
                                              eot_embd)
    # adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)

    # replace_min_id = embedding_to_input_id(adv_embedding_min, all_embeddings.squeeze(0)).to(sot_min_id.device)
    # adv_embedding_min.data = id2embedding(tokenizer, all_embeddings, replace_min_id.to(devices[0]),
    #                                       devices[0])

    input_adv_word_embedding_min = construct_embd(k_min, adv_embedding_min, attack_type, sot_min_embd, mid_min_embd,
                                                  eot_min_embd)
    # adv_input_min_ids = construct_id(k_min, replace_min_id, attack_type, sot_min_id, eot_min_id, mid_min_id)

    return input_adv_word_embedding, adv_input_ids, input_adv_word_embedding_min, adv_input_min_ids


# def soft_prompt_attack_max(global_step, word,retain_word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p,
#                                start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,
#                                attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None,
#                                attack_init_embd=None, attack_method='pgd'):
#         epsilon = 5 / 100
#         #     '''
#         #     Perform soft prompt attack on the ESD model
#         #     Args:
#         #         attack_type: str
#         #             The type of attack (add or insert)
#         #         attack_embd_type: str
#         #             The type of adversarial embedding (condition_embd or word_embd)
#         #         attack_step: int
#         #             The number of steps for the attack
#         #         attack_lr: float
#         #             The learning rate for the attack
#         #         attack_init: str
#         #             The initialization method for the attack (latest or random)
#         #         attack_init_embd: torch.Tensor
#         #             The initial adversarial embedding
#         #     '''
#         try:
#             orig_prompt_len = len(word.split())
#         except:
#             orig_prompt_len = len(word)
#         if attack_type == 'add':
#             k = orig_prompt_len
#
#         # print(f'======== Attack Round {attack_round} ========')
#         quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
#                                                                  x, image_size, image_size, ddim_steps, s, ddim_eta,
#                                                                  start_code=code, till_T=t, verbose=False)
#
#         # Word Tokenization
#         text_input = tokenizer(
#             word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
#         )
#         k_max = 75
#         sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k_max - 1, orig_prompt_len)
#
#         # Word embedding for the prompt
#
#         text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
#
#         retain_input = tokenizer(
#             retain_word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
#         )
#         k_max = 75
#
#         # Word embedding for the prompt
#
#         retain_embeddings = id2embedding(tokenizer, all_embeddings, retain_input.input_ids.to(devices[0]), devices[0])
#
#
#
#         print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
#
#         bs,length,dim = text_embeddings.size(0),text_embeddings.size(1),text_embeddings.size(2)
#         beta = np.random.beta(1, 1, (bs, length, dim))
#         beta = torch.from_numpy(beta).to(devices[0]).float()
#
#         for i in range(attack_step):
#             beta.requires_grad = True
#             input_adv_word_embedding=beta*text_embeddings+(1-beta)*retain_embeddings
#             input_adv_word_embedding = torch.min(torch.max(input_adv_word_embedding, text_embeddings - epsilon),
#                                                  text_embeddings + epsilon)
#             # ===== Randomly sample a time step from 0 to 1000 =====
#             t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
#             og_num = round((int(t_enc) / ddim_steps) * 1000)
#             og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
#             t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
#             start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # random inital noise
#
#             with torch.no_grad():
#                 # generate an image with the concept from ESD model
#                 z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
#                                         int(t_enc))  # emb_p seems to work better instead of emb_0
#                 # get conditional and unconditional scores from frozen model at time step t and image z
#
#                 e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
#             # breakpoint()
#
#             # Construct input_ids and input_embeds for the ESD model
#             if attack_embd_type == 'word_embd':
#                 adv_input_ids = construct_id_max(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#                 input_adv_condition_embedding = \
#                     text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]
#
#             # get conditional score from ESD model with adversarial condition embedding
#             with torch.no_grad():
#                 # generate an image with the concept from ESD model
#                 z_adv = quick_sample_till_t(input_adv_condition_embedding.to(devices[0]), start_guidance, start_code,
#                                         int(t_enc))
#             e_n = model.apply_model(z_adv.to(devices[0]), t_enc_ddpm.to(devices[0]),
#                                     input_adv_condition_embedding.to(devices[0]))
#
#             e_p.requires_grad = False
#
#             # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
#             loss = criteria(e_n.to(devices[0]), e_p.to(devices[0]))
#             # loss=torch.cosine_similarity(e_n.to(devices[0]).view(bs,-1), e_p.to(devices[0]).view(bs,-1), dim=-1).mean()
#             # loss=F.kl_div(F.log_softmax(e_n.to(devices[0]),dim=1),F.softmax(e_p.to(devices[0]),dim=1),reduction='sum')
#             # loss.backward()
#
#             grad_na = torch.autograd.grad(loss, [beta])[0]
#             if attack_method == 'pgd':
#                 beta = beta.detach() + attack_lr * grad_na.detach()
#             elif attack_method == 'fast_at':
#                 beta = beta.detach() + attack_lr * torch.sign(grad_na.detach())
#         input_adv_word_embedding = beta*text_embeddings+(1-beta)*retain_embeddings
#         input_adv_word_embedding = torch.min(torch.max(input_adv_word_embedding, text_embeddings - epsilon),
#                                              text_embeddings + epsilon)
#         return input_adv_word_embedding, adv_input_ids

# def soft_prompt_attack(global_step, word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p,
#                        start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,
#                        attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None,
#                        attack_init_embd=None, attack_method='pgd'):
#     '''
#     Perform soft prompt attack on the ESD model
#     Args:
#         attack_type: str
#             The type of attack (add or insert)
#         attack_embd_type: str
#             The type of adversarial embedding (condition_embd or word_embd)
#         attack_step: int
#             The number of steps for the attack
#         attack_lr: float
#             The learning rate for the attack
#         attack_init: str
#             The initialization method for the attack (latest or random)
#         attack_init_embd: torch.Tensor
#             The initial adversarial embedding
#     '''
#     orig_prompt_len = len(word.split())
#     if attack_type == 'add':
#         k = orig_prompt_len
#
#     # print(f'======== Attack Round {attack_round} ========')
#     quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
#                                                              x, image_size, image_size, ddim_steps, s, ddim_eta,
#                                                              start_code=code, till_T=t, verbose=False)
#
#     # Word Tokenization
#     text_input = tokenizer(
#         word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
#     )
#     sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k, orig_prompt_len)
#
#     # Word embedding for the prompt
#     text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
#     sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)
#
#     if attack_init == 'latest':
#         adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1, attack_init_embd)
#     elif attack_init == 'random':
#         adv_embedding = init_adv(k, tokenizer, all_embeddings, attack_type, devices[0], 1)
#
#     attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)
#
#     if attack_embd_type == 'condition_embd':
#         input_adv_condition_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
#         adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#
#     print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
#     for i in range(attack_step):
#         # ===== Randomly sample a time step from 0 to 1000 =====
#         t_enc = torch.randint(ddim_steps, (1,), device=devices[0])  # time step from 1000 to 0 (0 being good)
#         og_num = round((int(t_enc) / ddim_steps) * 1000)
#         og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
#         t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
#         start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # random inital noise
#
#         with torch.no_grad():
#             # generate an image with the concept from ESD model
#             z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
#                                     int(t_enc))  # emb_p seems to work better instead of emb_0
#             # get conditional and unconditional scores from frozen model at time step t and image z
#             e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
#             e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
#         # breakpoint()
#
#         # Construct input_ids and input_embeds for the ESD model
#         if attack_embd_type == 'word_embd':
#             input_adv_word_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
#             adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
#             input_adv_condition_embedding = \
#             text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]
#
#         # get conditional score from ESD model with adversarial condition embedding
#         e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]),
#                                 input_adv_condition_embedding.to(devices[0]))
#         e_0.requires_grad = False
#         e_p.requires_grad = False
#
#         # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
#         loss = -criteria(e_n.to(devices[0]), e_p.to(devices[0]))
#         loss.backward()
#
#         if attack_method == 'pgd':
#             attack_opt.step()
#         elif attack_method == 'fast_at':
#             adv_embedding.grad.sign_()
#             attack_opt.step()
#         else:
#             raise ValueError('attack_method must be either pgd or fast_at')
#
#
#     if attack_embd_type == 'condition_embd':
#         return input_adv_condition_embedding, adv_input_ids
#     elif attack_embd_type == 'word_embd':
#         return input_adv_word_embedding, adv_input_ids
#     else:
#         raise ValueError('attack_embd_type must be either condition_embd or word_embd')


#     return input_adv_word_embedding,adv_input_ids


def split_embd(input_embed, k, orig_prompt_len):
    sot_embd, mid_embd, replace_embd, eot_embd = torch.split(input_embed,
                                                             [1, orig_prompt_len, k, 76 - orig_prompt_len - k], dim=1)
    return sot_embd, mid_embd, replace_embd, eot_embd


def split_id(input_ids, k, orig_prompt_len):
    sot_id, mid_id, replace_id, eot_id = torch.split(input_ids, [1, orig_prompt_len, k, 76 - orig_prompt_len - k],
                                                     dim=1)
    return sot_id, mid_id, replace_id, eot_id


def construct_embd_max(k, adv_embedding, replace_embd, sot_embd, mid_embd, eot_embd):
    embedding = torch.cat(
        [sot_embd, adv_embedding[:, 0:1] + mid_embd, replace_embd + adv_embedding[:, 1:k + 1], eot_embd], dim=1)
    return embedding


def construct_id_max(k, adv_id, insertion_location, sot_id, eot_id, mid_id):
    replace_id = eot_id[:, 0].repeat(1, mid_id.shape[1])
    input_ids = torch.cat([sot_id, adv_id, replace_id, eot_id], dim=1)
    return input_ids


def construct_embd(k, adv_embedding, insertion_location, sot_embd, mid_embd, eot_embd):
    if insertion_location == 'prefix_k':  # Prepend k words before the original prompt
        embedding = torch.cat([sot_embd, adv_embedding, mid_embd, eot_embd], dim=1)
    elif insertion_location == 'replace_k':  # Replace k words in the original prompt
        replace_embd = eot_embd[:, 0, :].repeat(1, mid_embd.shape[1], 1)
        embedding = torch.cat([sot_embd, adv_embedding, replace_embd, eot_embd], dim=1)
    elif insertion_location == 'add':  # Add perturbation to the original prompt
        replace_embd = eot_embd[:, 0, :].repeat(1, k, 1)
        embedding = torch.cat([sot_embd, adv_embedding + mid_embd, replace_embd, eot_embd], dim=1)
    elif insertion_location == 'suffix_k':  # Append k words after the original prompt
        embedding = torch.cat([sot_embd, mid_embd, adv_embedding, eot_embd], dim=1)
    elif insertion_location == 'mid_k':  # Insert k words in the middle of the original prompt
        embedding = [sot_embd, ]
        total_num = mid_embd.size(1)
        embedding.append(mid_embd[:, :total_num // 2, :])
        embedding.append(adv_embedding)
        embedding.append(mid_embd[:, total_num // 2:, :])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding, dim=1)
    elif insertion_location == 'insert_k':  # seperate k words into the original prompt with equal intervals
        embedding = [sot_embd, ]
        total_num = mid_embd.size(1)
        internals = total_num // (k + 1)
        for i in range(k):
            embedding.append(mid_embd[:, internals * i:internals * (i + 1), :])
            embedding.append(adv_embedding[:, i, :].unsqueeze(1))
        embedding.append(mid_embd[:, internals * (i + 1):, :])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding, dim=1)

    elif insertion_location == 'per_k_words':
        embedding = [sot_embd, ]
        for i in range(adv_embedding.size(1) - 1):
            embedding.append(adv_embedding[:, i, :].unsqueeze(1))
            embedding.append(mid_embd[:, 3 * i:3 * (i + 1), :])
        embedding.append(adv_embedding[:, -1, :].unsqueeze(1))
        embedding.append(mid_embd[:, 3 * (i + 1):, :])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding, dim=1)
    return embedding


def construct_id(k, adv_id, insertion_location, sot_id, eot_id, mid_id):
    if insertion_location == 'prefix_k':
        input_ids = torch.cat([sot_id, adv_id, mid_id, eot_id], dim=1)

    elif insertion_location == 'replace_k':
        replace_id = eot_id[:, 0].repeat(1, mid_id.shape[1])
        input_ids = torch.cat([sot_id, adv_id, replace_id, eot_id], dim=1)

    elif insertion_location == 'add':
        replace_id = eot_id[:, 0].repeat(1, k)
        input_ids = torch.cat([sot_id, mid_id, replace_id, eot_id], dim=1)

    elif insertion_location == 'suffix_k':
        input_ids = torch.cat([sot_id, mid_id, adv_id, eot_id], dim=1)

    elif insertion_location == 'mid_k':
        input_ids = [sot_id, ]
        total_num = mid_id.size(1)
        input_ids.append(mid_id[:, :total_num // 2])
        input_ids.append(adv_id)
        input_ids.append(mid_id[:, total_num // 2:])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids, dim=1)

    elif insertion_location == 'insert_k':
        input_ids = [sot_id, ]
        total_num = mid_id.size(1)
        internals = total_num // (k + 1)
        for i in range(k):
            input_ids.append(mid_id[:, internals * i:internals * (i + 1)])
            input_ids.append(adv_id[:, i].unsqueeze(1))
        input_ids.append(mid_id[:, internals * (i + 1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids, dim=1)

    elif insertion_location == 'per_k_words':
        input_ids = [sot_id, ]
        for i in range(adv_id.size(1) - 1):
            input_ids.append(adv_id[:, i].unsqueeze(1))
            input_ids.append(mid_id[:, 3 * i:3 * (i + 1)])
        input_ids.append(adv_id[:, -1].unsqueeze(1))
        input_ids.append(mid_id[:, 3 * (i + 1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids, dim=1)
    return input_ids