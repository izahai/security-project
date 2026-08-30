<div align="center">
      
# <br> AEGIS: Adversarial Target-Guided Retention-Data-Free Robust Concept Erasure from Diffusion Models

<div align="left">

Our proposed robust Concept Erasure framework, AEGIS, enhances diffusion models' safety by robustly erasing unwanted concepts through adversarial training, achieving an optimal balance between concept erasure and image generation quality.

## Prepare

### Environment Setup
A suitable conda environment named ```AEGIS``` can be created and activated with:

```
conda env create -f environment.yaml
conda activate AEGIS
```

### Files Download
* Base model - SD v1.4: download it from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt), and move it to ```models/sd-v1-4-full-ema.ckpt```
* COCO-10k (for CLIP score and FID): you can extract the image subset from COCO dataset, or you can download it from [here](https://drive.google.com/file/d/1Qgm3nNhp6ykamszN_ZvofvuzjryTsPHB/view?usp=sharing). Then, move it to `data/imgs/coco_10k`

<br>

## Code Implementation

### Step 1: AEGIS [Train]

#### Hyperparameters: 
* Concept to be erased: `--prompt`    (e.g., 'nudity')
* Trainable module within DM: `--train_method`
* Attack generation strategy : `--attack_method`
* Number of attack steps for the adversarial prompt generation: `--attack_step`
* Adversarial prompting strategy: `--attack_type`  ('prefix_k', 'replace_k' ,'add')

#### Command Example: AEGIS Training
```
python train-scripts/AEGIS.py --attack_init random --attack_step 1 --prompt 'nudity' --train_method 'full'
```

### Step 2: Attack Evaluation [Robustness Evaluation] 
Follow the instruction in [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack) to implement attacks on DMs with ```AEGIS``` Unet for robustness evaluation.


### Step 3: Image Generation Quality Evaluation [Model Utility Evaluation]
Generate 10k images for FID & CLIP evaluation 

```
bash jobs/fid_10k_generate.sh
```  

Calculate FID & CLIP scores using [T2IBenchmark](https://github.com/boomb0om/text2image-benchmark)

```
bash jobs/tri_quality_eval.sh
```   





