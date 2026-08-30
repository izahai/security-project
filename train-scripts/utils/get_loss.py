import torch

from .util import *
import torch.nn.functional as F

def get_train_loss(model, model_orig, text_encoder, sampler, emb_0, emb_p, emb_n, start_guidance, negative_guidance,
                   devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_embd=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model

        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)

        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model

        devices: list of devices for ESD and DDPM models 
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler

        criteria: loss function for ESD model

        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                             x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                             start_code=code, till_T=t, verbose=False)

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        # generate an image with the concept from ESD model
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code,
                                int(t_enc))  # emb_p seems to work better instead of emb_0
        # get conditional and unconditional scores from frozen model at time step t and image z
        e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
        e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))

    if adv_embd is None:
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
    else:
        if attack_embd_type == 'condition_embd':
            # Train with adversarial conditional embedding
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), adv_embd.to(devices[0]))
        elif attack_embd_type == 'word_embd':
            # Train with adversarial word embedding
            print('====== Training with adversarial word embedding =====')
            adv_emb_n = text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), adv_emb_n.to(devices[0]))
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')

    e_0.requires_grad = False
    e_p.requires_grad = False

    # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
    loss = criteria(e_n.to(devices[0]),
                    e_0.to(devices[0]) - (negative_guidance * (e_p.to(devices[0]) - e_0.to(devices[0]))))

    return loss


def get_train_loss_batch(unlearn_batch, retain_batch, retain_train, retain_loss_w, model, model_orig, text_encoder,
                         sampler, unlearn_emb_0, unlearn_emb_p, retain_emb_p, unlearn_emb_n, retain_emb_n,
                         start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria,
                         adv_input_ids, attack_embd_type, adv_embd=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model

        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)

        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model

        devices: list of devices for ESD and DDPM models 
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler

        criteria: loss function for ESD model

        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                    x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                    start_code=code, n_samples=batch, till_T=t,
                                                                    verbose=False)

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    # random nosie initialization
    unlearn_start_code = torch.randn((unlearn_batch, 4, 64, 64)).to(devices[0])
    if retain_train == 'reg':
        retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        # generate an image with the concept from ESD model
        unlearn_z = quick_sample_till_t(unlearn_emb_p.to(devices[0]), start_guidance, unlearn_start_code, unlearn_batch,
                                        int(t_enc))  # emb_p seems to work better instead of emb_0
        # get conditional and unconditional scores from frozen model at time step t and image z
        unlearn_e_0 = model_orig.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                             unlearn_emb_0.to(devices[0]))
        unlearn_e_p = model_orig.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                             unlearn_emb_p.to(devices[0]))

        if retain_train == 'reg':
            retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch,
                                           int(t_enc))  # emb_p seems to work better instead of emb_0
            # retain_e_0 = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_0.to(devices[0]))
            retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                                retain_emb_p.to(devices[0]))

    if adv_embd is None:
        unlearn_e_n = model.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                        unlearn_emb_n.to(devices[0]))
    else:
        if attack_embd_type == 'condition_embd':
            # Train with adversarial conditional embedding
            unlearn_e_n = model.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                            adv_embd.to(devices[0]))
        elif attack_embd_type == 'word_embd':
            # Train with adversarial word embedding
            print('====== Training with adversarial word embedding =====')
            adv_emb_n = text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            unlearn_e_n = model.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                            adv_emb_n.to(devices[0]))
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')

    unlearn_e_0.requires_grad = False
    unlearn_e_p.requires_grad = False
    if retain_train == 'reg':
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        unlearn_loss = criteria(unlearn_e_n.to(devices[0]), unlearn_e_0.to(devices[0]) - (
                    negative_guidance * (unlearn_e_p.to(devices[0]) - unlearn_e_0.to(devices[0]))))

        retain_e_n = model.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_n.to(devices[0]))

        # retain_e_0.requires_grad = False
        retain_e_p.requires_grad = False
        retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))

        loss = unlearn_loss + retain_loss_w * retain_loss
        return loss

    else:
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        unlearn_loss = criteria(unlearn_e_n.to(devices[0]), unlearn_e_0.to(devices[0]) - (
                    negative_guidance * (unlearn_e_p.to(devices[0]) - unlearn_e_0.to(devices[0]))))
        return unlearn_loss


def get_ca_train_loss_batch(unlearn_batch, retain_batch, retain_train, retain_loss_w, model, model_orig, text_encoder,
                            sampler, unlearn_emb_0, unlearn_emb_p, retain_emb_p, unlearn_emb_n, retain_emb_n,
                            retaining_emb_p, retaining_emb_n, start_guidance, negative_guidance, devices, ddim_steps,
                            ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_embd=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model

        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)

        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model

        devices: list of devices for ESD and DDPM models 
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler

        criteria: loss function for ESD model

        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                    x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                    start_code=code, n_samples=batch, till_T=t,
                                                                    verbose=False)

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    # random nosie initialization
    unlearn_start_code = torch.randn((unlearn_batch, 4, 64, 64)).to(devices[0])

    retaining_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        # generate an image with the concept from ESD model
        unlearn_z = quick_sample_till_t(unlearn_emb_p.to(devices[0]), start_guidance, unlearn_start_code, unlearn_batch,
                                        int(t_enc))  # emb_p seems to work better instead of emb_0
        # get conditional and unconditional scores from frozen model at time step t and image z
        # unlearn_e_0 = model_orig.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]), unlearn_emb_0.to(devices[0]))
        unlearn_e_p = model_orig.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                             unlearn_emb_p.to(devices[0]))

        # retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
        # retain_e_0 = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_0.to(devices[0]))
        retain_z = unlearn_z
        retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                            retain_emb_p.to(devices[0]))

        if retain_train == 'reg':
            retaining_z = quick_sample_till_t(retaining_emb_p.to(devices[0]), start_guidance, retaining_start_code,
                                              retain_batch, int(t_enc))  # emb_p seems to work better instead of emb_0
            # retain_e_0 = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_0.to(devices[0]))
            retaining_e_p = model_orig.apply_model(retaining_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                                   retaining_emb_p.to(devices[0]))

    if adv_embd is None:
        unlearn_e_n = model.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                        unlearn_emb_n.to(devices[0]))
    else:
        if attack_embd_type == 'condition_embd':
            # Train with adversarial conditional embedding
            unlearn_e_n = model.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                            adv_embd.to(devices[0]))
        elif attack_embd_type == 'word_embd':
            # Train with adversarial word embedding
            print('====== Training with adversarial word embedding =====')
            adv_emb_n = text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            unlearn_e_n = model.apply_model(unlearn_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                            adv_emb_n.to(devices[0]))
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')

    # unlearn_e_0.requires_grad = False
    unlearn_e_p.requires_grad = False
    retain_e_p.requires_grad = False

    # ca_loss = criteria(unlearn_e_n.to(devices[0]), retain_e_p.to(devices[0]))

    # return ca_loss

    if retain_train == 'reg':
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        ca_loss = criteria(unlearn_e_n.to(devices[0]), retain_e_p.to(devices[0]))

        retaining_e_n = model.apply_model(retaining_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                          retaining_emb_n.to(devices[0]))

        # retain_e_0.requires_grad = False
        retaining_e_p.requires_grad = False
        retain_loss = criteria(retaining_e_n.to(devices[0]), retaining_e_p.to(devices[0]))

        loss = ca_loss + retain_loss_w * retain_loss
        return loss

    else:
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        ca_loss = criteria(unlearn_e_n.to(devices[0]), retain_e_p.to(devices[0]))
        return ca_loss


def Gradient_R(parameters,parameters_orig,loss1, retain_w, step,w,mu=0.5):
    for param in parameters:
        if param.requires_grad:
            param.grad = None
    P_gradient1 = torch.autograd.grad(loss1, parameters, create_graph=True)

    for param in parameters:
        if param.requires_grad:
            param.grad = None

    # P_gradient2 = torch.autograd.grad(loss2, parameters, create_graph=True,retain_graph=True)
    P_gradient2=[]
    for param1,param2 in zip(parameters,parameters_orig):
        P_gradient2.append((param1-param2)*retain_w)

    i = 0

    # cos_list = []
    for (p_w1, p_w2) in zip(P_gradient1, P_gradient2):
        grad1 = torch.flatten(p_w1.data)
        grad2 = torch.flatten(p_w2.data)

        # cos = torch.cosine_similarity(grad1.view(-1), grad2.view(-1), dim=0)
        # cos_list.append(cos.view(-1))
        if torch.sum(grad1 * grad2) >= 0:
            parameters[i].grad = p_w1.data
        else:
            grad_proj_flatten = grad2 * torch.mm(grad1.view(1, -1), grad2.view(-1, 1)) / (
                    torch.mm(grad2.view(1, -1), grad2.view(-1, 1)) + 1e-20)
            grad_proj = p_w2 * torch.mm(grad1.view(1, -1), grad2.view(-1, 1)) / (
                    torch.mm(grad2.view(1, -1), grad2.view(-1, 1)) + 1e-20)
            delta_w = torch.norm(torch.mm(grad1.view(1, -1), grad_proj_flatten.view(-1, 1)), p=2)
            if step == 0:
                w[i] = max(0, min(delta_w, 1))
             
            else:
                try:
                    w[i] = max(0, min(delta_w * (1 - mu) + w[i] * mu, 1))
                except:
                    w[i] = max(0, min(delta_w, 1))
            g_step = p_w1 - w[i] * grad_proj

            parameters[i].grad = g_step.reshape_as(parameters[i])
        i += 1


def get_train_loss_retain(global_step,w,retain_batch, retain_train, retain_loss_w, parameters,parameters_orig, model, model_orig, text_encoder,
                            sampler, emb_0,emb_new, emb_p, retain_emb_p, emb_n, retain_emb_n, start_guidance, negative_guidance,
                            devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type,
                            adv_embd=None, adv_embd_Min=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model

        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)

        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model

        devices: list of devices for ESD and DDPM models
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler

        criteria: loss function for ESD model

        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                    x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                    start_code=code, n_samples=batch, till_T=t,
                                                                    verbose=False)

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
    # retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        # generate an image with the concept from ESD model
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, 1,
                                int(t_enc))  # emb_p seems to work better instead of emb_0

        # retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch,
        #                                int(t_enc))
        # get conditional and unconditional scores from frozen model at time step t and image z
        e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
        e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        e_new = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_new.to(devices[0]))
        # retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
        #                                     retain_emb_p.to(devices[0]))

        #
    # if adv_embd is None:

    e_0.requires_grad = False
    e_p.requires_grad = False
    # retain_e_p.requires_grad = False


    e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
    # retain_e_n = model.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_n.to(devices[0]))

    unlearn_loss = criteria(e_n.to(devices[0]),
                                e_0.to(devices[0]) - negative_guidance *(e_p.to(devices[0])-e_0.to(devices[0])))
    # - (negative_guidance * (e_p.to(devices[0])))
    # - e_0.to(devices[0]).detach()
    # retain_loss = criteria(retain_n_pred.to(devices[0]), retain_p_pred.to(devices[0]))
    # retain_loss=F.kl_div(F.log_softmax(retain_e_n.view(retain_batch,-1).to(devices[0]), dim=1),F.softmax(retain_e_p.view(retain_batch,-1).to(devices[0]), dim=1),reduction='batchmean')
        #
        #
        #
    # Gradient_R(parameters,parameters_orig,unlearn_loss, retain_loss_w,global_step,w)
    total_loss = unlearn_loss
        # unlearn_loss = criteria(e_n_mix.to(devices[0]), e_0.to(devices[0]))
        # unlearn_loss=criteria(e_n_mix.to(devices[0]), e_0.to(devices[0]))+criteria(e_n.to(devices[0]), e_0.to(devices[0]))
    return unlearn_loss


def Param_Gap(parameters,parameters_orig,criteria):

    retain_loss=0.
    for param1,param2 in zip(parameters,parameters_orig):
        retain_loss += criteria(param1,param2)
    return retain_loss

def get_train_loss_retain_Pareto(global_step,w,retain_batch, retain_train, retain_loss_w, parameters,parameters_orig, model, model_orig, text_encoder,
                            sampler, emb_0,emb_new, emb_p, retain_emb_p, emb_n, retain_emb_n, start_guidance, negative_guidance,
                            devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type,
                            adv_embd=None, adv_embd_Min=None,nesh_opt=None,opt=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model

        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)

        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model

        devices: list of devices for ESD and DDPM models
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler

        criteria: loss function for ESD model

        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                    x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                    start_code=code, n_samples=batch, till_T=t,
                                                                    verbose=False)

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
    retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        # generate an image with the concept from ESD model
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, 1,
                                int(t_enc))  # emb_p seems to work better instead of emb_0

        retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch,
                                       int(t_enc))
        # get conditional and unconditional scores from frozen model at time step t and image z
        e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
        e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        e_new = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_new.to(devices[0]))
        retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]),
                                            retain_emb_p.to(devices[0]))

        #
    # if adv_embd is None:

    e_0.requires_grad = False
    e_p.requires_grad = False
    retain_e_p.requires_grad = False

    e_n_mix = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
    retain_e_n = model.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_p.to(devices[0]))


    unlearn_loss = criteria(e_n_mix.to(devices[0]),
                                e_new.to(devices[0]) - (negative_guidance * (e_p.to(devices[0]) - e_0.to(devices[0]))))

    retain_loss = F.kl_div(F.log_softmax(retain_e_n.view(retain_batch, -1).to(devices[0]), dim=1),
                           F.softmax(retain_e_p.view(retain_batch, -1).to(devices[0]), dim=1), reduction='batchmean')

    # retain_loss = criteria(e_n.to(devices[0]), retain_e_p.to(devices[0]))
    # retain_loss=Param_Gap(parameters,parameters_orig,criteria)
        #
        #
        #
    # Gradient_R(parameters,unlearn_loss, retain_loss_w * retain_loss,global_step,w)
    loss, _ = nesh_opt.backward(opt, [unlearn_loss, retain_loss], parameters)
    # loss = unlearn_loss + retain_loss_w * retain_loss
        # unlearn_loss = criteria(e_n_mix.to(devices[0]), e_0.to(devices[0]))
        # unlearn_loss=criteria(e_n_mix.to(devices[0]), e_0.to(devices[0]))+criteria(e_n.to(devices[0]), e_0.to(devices[0]))
    return loss