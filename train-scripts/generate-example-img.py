from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from utils.text_encoder import CustomTextEncoder

def get_openai_diffuser_transformer(diffuser_ckpt):
  open_ckpt = {}
  for i in range(12):
    open_ckpt[f'transformer.resblocks.{i}.ln_1.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm1.weight']
    open_ckpt[f'transformer.resblocks.{i}.ln_1.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm1.bias']
    
    q_proj_weight = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.q_proj.weight']
    k_proj_weight = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.k_proj.weight']
    v_proj_weight = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.v_proj.weight']
    proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    q_proj_bias = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.q_proj.bias']
    k_proj_bias = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.k_proj.bias']
    v_proj_bias = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.v_proj.bias']
    proj_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
    open_ckpt[f'transformer.resblocks.{i}.attn.in_proj_weight'] = proj_weight
    open_ckpt[f'transformer.resblocks.{i}.attn.in_proj_bias'] = proj_bias
    
    open_ckpt[f'transformer.resblocks.{i}.attn.out_proj.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.out_proj.weight']
    open_ckpt[f'transformer.resblocks.{i}.attn.out_proj.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.out_proj.bias']
    
    open_ckpt[f'transformer.resblocks.{i}.ln_2.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm2.weight']
    open_ckpt[f'transformer.resblocks.{i}.ln_2.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm2.bias']
    open_ckpt[f'transformer.resblocks.{i}.mlp.c_fc.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc1.weight']
    open_ckpt[f'transformer.resblocks.{i}.mlp.c_fc.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc1.bias']
    open_ckpt[f'transformer.resblocks.{i}.mlp.c_proj.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc2.weight']
    open_ckpt[f'transformer.resblocks.{i}.mlp.c_proj.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc2.bias']

  open_ckpt['ln_final.weight'] = diffuser_ckpt['text_model.final_layer_norm.weight']
  open_ckpt['ln_final.bias'] = diffuser_ckpt['text_model.final_layer_norm.bias']
  
  return open_ckpt



def extract_text_encoder_ckpt(ckpt_path):
    full_ckpt = torch.load(ckpt_path, weights_only=False)
    new_ckpt = {}
    for key in full_ckpt.keys():
        if 'text_encoder.text_model' in key:
            new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
    return new_ckpt

def generate_images(model_name, prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0, folder_suffix='imagenette', origin_or_target='target', gen_batch_size=16, fp16=False):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)
    
    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    '''
    if model_name == 'SD-v1-4':
        dir_ = "CompVis/stable-diffusion-v1-4"
    elif model_name == 'SD-V2':
        dir_ = "stabilityai/stable-diffusion-2-base"
    elif model_name == 'SD-V2-1':
        dir_ = "stabilityai/stable-diffusion-2-1-base"
    else:
        dir_ = "CompVis/stable-diffusion-v1-4" # all the erasure models built on SDv1-4
    
    if origin_or_target == 'target':  
        target_ckpt = f'{save_path}.pt'
        save_path = f'{save_path}_visualizations_{folder_suffix}'
    else:
        target_ckpt = None
        save_path = f'{save_path}/original_SD/visualizations_{folder_suffix}'
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")
    # if 'SD' not in model_name:
    #     try:
    #         model_path = f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
    #         unet.load_state_dict(torch.load(model_path))
    #     except Exception as e:
    #         print(f'Model path is not valid, please check the file name and structure: {e}')
    if origin_or_target == 'target':
        if 'TextEncoder' not in target_ckpt:
            unet.load_state_dict(torch.load(target_ckpt))
        else:
            text_encoder.load_state_dict(extract_text_encoder_ckpt(target_ckpt), strict=False)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # fp16 halves VRAM and ~2x throughput on tensor cores. Cast after loading the
    # fp32 target ckpt so load_state_dict doesn't upcast the module back to fp32.
    dtype = torch.float16 if fp16 else torch.float32
    if fp16:
        vae = vae.half(); text_encoder = text_encoder.half(); unet = unet.half()
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)

    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    height = image_size
    width = image_size
    max_length = tokenizer.model_max_length
    scheduler.set_timesteps(ddim_steps)

    from tqdm.auto import tqdm

    # Flatten to one job per image. The latent is drawn here with the row's own
    # seed (bit-identical to the per-row generation above) so batching across
    # prompts changes nothing except throughput. Skip images already on disk so
    # a killed run resumes for free.
    jobs = []
    for _, row in df.iterrows():
        case_number = int(row.case_number)
        if case_number < from_case:
            continue
        gen = torch.manual_seed(int(row.evaluation_seed))
        lats = torch.randn((num_samples, unet.in_channels, height // 8, width // 8), generator=gen)
        for s in range(num_samples):
            if os.path.exists(f"{folder_path}/{case_number}_{s}.png"):
                continue
            jobs.append((case_number, s, str(row.prompt), lats[s]))

    for start in tqdm(range(0, len(jobs), gen_batch_size)):
        batch = jobs[start:start + gen_batch_size]
        prompts = [b[2] for b in batch]
        latents = torch.stack([b[3] for b in batch]).to(torch_device).to(dtype)

        text_input = tokenizer(prompts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        uncond_input = tokenizer([""] * len(batch), padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        for (case_number, s, _, _), im in zip(batch, images):
            Image.fromarray(im).save(f"{folder_path}/{case_number}_{s}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=False, default='SD-v1-4')
    # parser.add_argument('--target_ckpt', help='path to unlearned CKPT', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=False, default='./data/prompts/visualization_example.csv')
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
    parser.add_argument('--folder_suffix', help='folder dir suffix', type=str, required=True)
    parser.add_argument('--origin_or_target', help='origin or target', type=str, required=False, default='target')
    parser.add_argument('--batch_size', help='images generated per forward pass', type=int, required=False, default=8)
    parser.add_argument('--fp16', help='run generation in float16 (faster, less VRAM)', action='store_true')
    args = parser.parse_args()
    
    model_name = args.model_name
    # target_ckpt = args.target_ckpt
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    folder_suffix = args.folder_suffix
    origin_or_target = args.origin_or_target
    
    generate_images(model_name, prompts_path, save_path, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, folder_suffix = folder_suffix, origin_or_target=origin_or_target, gen_batch_size=args.batch_size, fp16=args.fp16)
