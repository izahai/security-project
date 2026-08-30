from utils.util import *
from utils.attack_util import *
from utils.text_encoder import CustomTextEncoder
from utils.get_loss import *
from utils.prompt_dataset import *

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import random
import argparse
import wandb
from pathlib import Path
import os

wandb.login(key='')

class GP:
    def __init__(self, parameter_orig, mu=0.2):
        self.parameter_orig = parameter_orig
        self.mu = mu
        self.pre_g_hat = []
        self.t = 0
        self.w = []
        self.w_init=0.0
        self.beta1=0.9
        self.beta2=0.999
        self.m=[]
        self.v=[]

    def DGR(self, parameter,lr,t):
        index = 0
        cos=0.
        w=0.
        for para1, para2 in zip(parameter, self.parameter_orig):

            if para1.grad is not None:
                g_e = para1.grad.data
                g_r =2.*(para1.data - para2.data)
                # para1.grad.data = g_e+g_r
                grad_e = torch.flatten(g_e)
                grad_r = torch.flatten(g_r)
                cos += torch.cosine_similarity(grad_e, grad_r, dim=-1).item()
                grad_r_norm = torch.norm(grad_r, p=2)
                dgr_w=torch.dot(grad_e, grad_r) / (grad_r_norm**2)
                grad_proj_flatten = grad_r * dgr_w
                grad_proj = g_r * dgr_w
                if torch.sum(grad_e * grad_r) < 0:
                    if self.t != 0:
                        tmp1=torch.dot(grad_e,self.pre_g_hat[index])
                        delta_w = torch.sign(tmp1).detach()
                        new_w=max(self.w_init, min(self.w[index]-self.mu*delta_w,1e6))
                        self.w[index] = new_w
                        w+=new_w
                        self.pre_g_hat[index]=grad_proj_flatten.detach()
                    else:
                        tmp1 = torch.dot(grad_e, grad_proj_flatten)
                        delta_w = torch.sign(tmp1).detach()
                        self.m.append(0.)
                        self.v.append(0.)
                        new_w=max(self.w_init, min(self.w_init-self.mu*delta_w,1e6))
                        self.w.append(new_w)
                        self.pre_g_hat.append(grad_proj_flatten.detach())
                        w+=new_w
                    g_step = g_e-self.w[index] * grad_proj
                    # g_step = g_e +grad_proj
                else:
                    if self.t != 0:
                        tmp1=torch.dot(grad_e, self.pre_g_hat[index])
                        # tmp1=grad_e*self.pre_g_hat[index]
                        delta_w = torch.sign(tmp1).detach()
                        new_w=max(self.w_init, min(self.w[index]-self.mu*delta_w,1e6))
                        self.w[index] = new_w
                        w+=new_w
                        self.pre_g_hat[index] = grad_proj_flatten.detach()
                        print(delta_w.item())
                    else:
                        tmp1 = torch.dot(grad_e, grad_proj_flatten)
                        delta_w = torch.sign(tmp1).detach()
                        new_w = max(self.w_init, min(self.w_init - self.mu * delta_w, 1e6))
                        w+=new_w
                        self.w.append(new_w)
                        self.pre_g_hat.append(grad_proj_flatten.detach())
                    g_step = g_e
                para1.grad.data = g_step.reshape_as(para1.data)
                index += 1
        self.t+=1
        return cos/index,w/index
def AEGIS(prompt, dataset_retain, retain_batch, retain_train, retain_step, retain_loss_w, attack_method,
               train_method, norm_layer, component, start_guidance, negative_guidance, iterations, save_interval, lr,
               config_path, ckpt_path, diffusers_config_path, output_dir, devices, seperator=None, image_size=512,
               ddim_steps=50, adv_prompt_num=3, attack_embd_type='word_embd', attack_type='prefix_k',
               attack_init='latest', warmup_iter=200, attack_step=30, attack_lr=1e-2, adv_prompt_update_step=20):
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                    x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                    start_code=code, n_samples=batch, till_T=t,
                                                                    verbose=False)

    # ====== Stage 0: PROMPT CLEANING ======
    word_print = prompt.replace(' ', '')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(f'The Concept Prompt to be unlearned:{words}')

    retain_dataset = retain_prompt(dataset_retain)

    # ======= Stage 1: TRAINING SETUP =======
    ddim_eta = 0
    model_name_or_path = "CompVis/stable-diffusion-v1-4"
    cache_path = ".cache"

    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer", cache_dir=cache_path)
    text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder", cache_dir=cache_path).to(
        devices[0])
    custom_text_encoder = CustomTextEncoder(text_encoder).to(devices[0])
    custom_text_encoder_orig = CustomTextEncoder(text_encoder).to(devices[0])

    all_embeddings = custom_text_encoder.get_all_embedding().unsqueeze(0)

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)
    model_orig.eval()
    custom_text_encoder_orig.eval()

    # Setup tainable model parameters
    if 'text_encoder' in train_method:
        parameters = param_choices(model=custom_text_encoder, train_method=train_method, component=component,
                                   final_layer_norm=norm_layer)
        parameters_orig = param_choices(model=custom_text_encoder_orig, train_method=train_method, component=component,
                                        final_layer_norm=norm_layer)
    else:
        parameters = param_choices(model=model, train_method=train_method, component=component,
                                   final_layer_norm=norm_layer)

        parameters_orig = param_choices(model=model_orig, train_method=train_method, component=component,
                                        final_layer_norm=norm_layer)

    gradient_GP = GP(parameters_orig)
    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = train_method

    # ========== Stage 2: Training ==========
    pbar = tqdm(range(iterations))
    global_step = 0
    train_step = 0
    attack_round = 0
    w_m = {}
    for i in pbar:
        # Change unlearned concept and obtain its corresponding adv embedding
        if i % adv_prompt_update_step == 0:


            word = random.sample(words, 1)[0]
            text_input = tokenizer(
                word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
            )
            text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])

            # get conditional embedding for the prompt
            emb_0 = model_orig.get_learned_conditioning([''])
            emb_p = model_orig.get_learned_conditioning([word])

            # ===== ** Attack ** : get adversarial prompt

            if attack_round == 0:

                adv_word_embd, adv_input_ids, adv_word_embd_min, adv_input_ids_min = soft_prompt_attack_max(global_step,
                                                                                                            word, model,
                                                                                                            model_orig,
                                                                                                            tokenizer,
                                                                                                            custom_text_encoder,
                                                                                                            sampler,
                                                                                                            emb_0,
                                                                                                            emb_p,
                                                                                                            start_guidance,
                                                                                                            devices,
                                                                                                            ddim_steps,
                                                                                                            ddim_eta,
                                                                                                            image_size,
                                                                                                            criteria,
                                                                                                            adv_prompt_num,
                                                                                                            all_embeddings,
                                                                                                            attack_round,
                                                                                                            attack_type,
                                                                                                            attack_embd_type,
                                                                                                            1,
                                                                                                            attack_lr,
                                                                                                            attack_init,
                                                                                                            None, None,
                                                                                                            attack_method)


            else:

                adv_word_embd, adv_input_ids, adv_word_embd_min, adv_input_ids_min = soft_prompt_attack_max(global_step,
                                                                                                            word, model,
                                                                                                            model_orig,
                                                                                                            tokenizer,
                                                                                                            custom_text_encoder,
                                                                                                            sampler,
                                                                                                            emb_0,
                                                                                                            emb_p,
                                                                                                            start_guidance,
                                                                                                            devices,
                                                                                                            ddim_steps,
                                                                                                            ddim_eta,
                                                                                                            image_size,
                                                                                                            criteria,
                                                                                                            adv_prompt_num,
                                                                                                            all_embeddings,
                                                                                                            attack_round,
                                                                                                            attack_type,
                                                                                                            attack_embd_type,
                                                                                                            1,
                                                                                                            attack_lr,
                                                                                                            attack_init,
                                                                                                            adv_word_embd.detach(),
                                                                                                            adv_word_embd_min.detach(),
                                                                                                            attack_method)

            global_step += attack_step
            attack_round += 1

        # Set model/TextEnocder to train or eval mode
        if 'text_encoder' in train_method:
            custom_text_encoder.text_encoder.train()
            custom_text_encoder.text_encoder.requires_grad_(True)
            model.eval()
            # print('==== Train text_encoder ====')
        else:
            custom_text_encoder.text_encoder.eval()
            custom_text_encoder.text_encoder.requires_grad_(False)
            model.train()
        opt.zero_grad()

        retain_emb_p =None
        retain_emb_n=None
        input_ids = text_input.input_ids.to(devices[0])
        emb_new = custom_text_encoder_orig(input_ids=input_ids, inputs_embeds=text_embeddings)[0]
        emb_n = custom_text_encoder(input_ids=adv_input_ids_min, inputs_embeds=adv_word_embd_min)[0]
        # emb_n = custom_text_encoder(input_ids=input_ids, inputs_embeds=text_embeddings)[0]
        emb_p = custom_text_encoder_orig(input_ids=adv_input_ids, inputs_embeds=adv_word_embd)[0]
        # emb_p=custom_text_encoder_orig(input_ids=adv_input_ids_min, inputs_embeds=adv_word_embd_min)[0]

        loss = get_train_loss_retain(train_step, w_m, retain_batch, retain_train, retain_loss_w, parameters,
                                     parameters_orig, model,
                                     model_orig,
                                     custom_text_encoder, sampler, emb_0, emb_new, emb_p, retain_emb_p, emb_n,
                                     retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps,
                                     ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type,
                                     adv_word_embd)
        opt.zero_grad()
        loss.backward()

        cos,w=gradient_GP.DGR(parameters,lr,train_step)
        print(w)
        # loss=loss_e+loss_r
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        wandb.log({'Train_Loss': loss.item()}, step=global_step)
        wandb.log({'Cos': cos}, step=global_step)
        wandb.log({'W': w}, step=global_step)
        global_step += 1
        train_step += 1
        opt.step()

        # ====== Stage 3: save final model and loss curve ======
        # save checkpoint and loss curve
        if (i + 1) % save_interval == 0 and i + 1 != iterations and i + 1 >= save_interval:
            if 'text_encoder' in train_method:
                save_text_encoder(output_dir, custom_text_encoder, name, i)
            else:
                save_model(output_dir, model, name, i, save_compvis=True, save_diffusers=True,
                           compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

        # if i % 1 == 0:
        #     save_history(output_dir, losses, word_print)

    # Save final model and loss curve
    model.eval()
    custom_text_encoder.text_encoder.eval()
    custom_text_encoder.text_encoder.requires_grad_(False)
    if 'text_encoder' in train_method:
        save_text_encoder(output_dir, custom_text_encoder, name, i)
    else:
        save_model(output_dir, model, name, i, save_compvis=True, save_diffusers=True, compvis_config_file=config_path,
                   diffusers_config_file=diffusers_config_path)
    save_history(output_dir, losses, word_print)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AEGIS',
        description='Source Code of AEGIS')

    # Diffusion setup
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False,
                        default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float,
                        required=False, default=1)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str,
                        required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False,
                        default='models/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False,
                        default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str,
                        required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=50)

    # Training setup
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=False,
                        default='nudity')
    parser.add_argument('--dataset_retain', help='prompts corresponding to non-target concept to retain', type=str,
                        required=False, default='coco_object',
                        choices=['coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'])
    # parser.add_argument('--unlearn_batch', help='batch size of unlearning prompt during training', type=int, required=False, default=5)
    parser.add_argument('--retain_batch', help='batch size of retaining prompt during training', type=int,
                        required=False, default=5)
    parser.add_argument('--retain_train', help='different retaining version: reg (regularization) or iter (iterative)',
                        type=str, required=False, default='reg', choices=['iter', 'reg'])
    parser.add_argument('--retain_step', help='number of steps for retaining prompts', type=int, required=False,
                        default=1)
    parser.add_argument('--retain_loss_w', help='retaining loss weight', type=float, required=False, default=1.0)

    parser.add_argument('--train_method', help='method of training', type=str,
                        choices=['text_encoder_full', 'text_encoder_layer0', 'text_encoder_layer01',
                                 'text_encoder_layer012', 'text_encoder_layer0123', 'text_encoder_layer01234',
                                 'text_encoder_layer012345', 'text_encoder_layer0123456', 'text_encoder_layer01234567',
                                 'text_encoder_layer012345678', 'text_encoder_layer0123456789',
                                 'text_encoder_layer012345678910', 'text_encoder_layer01234567891011',
                                 'text_encoder_layer0_11', 'text_encoder_layer01_1011', 'text_encoder_layer012_91011',
                                 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer'],
                        default='noxattn', required=False)
    parser.add_argument('--norm_layer', help='During training, norm layer to be updated or not', action='store_true',
                        default=False, required=False)
    parser.add_argument('--attack_method', help='method of training', type=str,
                        choices=['pgd', 'multi_pgd', 'fast_at', 'free_at'], default='pgd', required=False)
    parser.add_argument('--component', help='component', type=str, choices=['all', 'ffn', 'attn'], default='all',
                        required=False)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--save_interval', help='iterations used to train', type=int, required=False, default=200)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)

    # Attack hyperparameters
    parser.add_argument('--adv_prompt_num', help='number of prompt token for adversarial soft prompt learning',
                        type=int, required=False, default=1)
    parser.add_argument('--attack_embd_type', help='the adversarial embd type: word embedding, condition embedding',
                        type=str, required=False, default='word_embd', choices=['word_embd', 'condition_embd'])
    parser.add_argument('--attack_type', help='the attack type: append or add', type=str, required=False,
                        default='prefix_k',
                        choices=['replace_k', 'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'])
    parser.add_argument('--attack_init', help='the attack init: random or latest', type=str, required=False,
                        default='latest', choices=['random', 'latest'])
    parser.add_argument('--attack_step', help='adversarial attack steps', type=int, required=False, default=10)
    parser.add_argument('--adv_prompt_update_step', help='after every n step, adv prompt would be updated', type=int,
                        required=False, default=1)
    parser.add_argument('--attack_lr', help='learning rate used to train', type=float, required=False, default=1e-3)
    parser.add_argument('--warmup_iter', help='the number of warmup interations before attack', type=int,
                        required=False, default=200)

    # Log details
    parser.add_argument('--project_name', help='wandb project name', type=str, required=False, default='AEGIS')

    args = parser.parse_args()

    prompt = args.prompt
    dataset_retain = args.dataset_retain
    retain_batch = args.retain_batch
    retain_train = args.retain_train
    retain_step = args.retain_step
    retain_loss_w = args.retain_loss_w

    train_method = args.train_method
    norm_layer = args.norm_layer
    attack_method = args.attack_method
    component = args.component
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    save_interval = args.save_interval
    lr = args.lr

    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    adv_prompt_num = args.adv_prompt_num
    attack_embd_type = args.attack_embd_type
    attack_type = args.attack_type
    attack_init = args.attack_init
    attack_step = args.attack_step
    attack_lr = args.attack_lr
    adv_prompt_update_step = args.adv_prompt_update_step
    warmup_iter = args.warmup_iter

    # Directory setup
    experiment_name = f'AEGIS'
    run_dir = Path("./results/results_with_AEGIS") / "wandb_logs"
    output_dir = Path("./results/results_with_AEGIS") / experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if not output_dir.exists():
        os.makedirs(str(output_dir))

    wandb.init(config=args,
               project=args.project_name,
               name=experiment_name,
               dir=str(run_dir),
               reinit=True)

    print_args_table(parser, args)

    AEGIS(prompt=prompt, dataset_retain=dataset_retain, retain_batch=retain_batch, retain_train=retain_train,
               retain_step=retain_step, retain_loss_w=retain_loss_w, attack_method=attack_method,
               train_method=train_method, norm_layer=norm_layer, component=component, start_guidance=start_guidance,
               negative_guidance=negative_guidance, iterations=iterations, save_interval=save_interval, lr=lr,
               config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path,
               output_dir=output_dir, devices=devices, seperator=seperator, image_size=image_size,
               ddim_steps=ddim_steps, adv_prompt_num=adv_prompt_num, attack_embd_type=attack_embd_type,
               attack_type=attack_type, attack_init=attack_init, warmup_iter=warmup_iter, attack_step=attack_step,
               attack_lr=attack_lr, adv_prompt_update_step=adv_prompt_update_step)
