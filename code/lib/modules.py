import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from lib.utils import truncated_noise
from lib.utils import mkdir_p, get_rank

from lib.datasets import TextImgDataset as Dataset
from lib.datasets import prepare_data, encode_tokens
from models.inception import InceptionV3
from pretrain_DAMSM_config import cfg, cfg_from_file
from GlobalAttention import func_attention
from utils import cosine_similarity
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
from lib.masks import mask_correlated_samples
from lib.nt_xent import NT_Xent

############   modules   ############
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def train(dataloader, netG, netD, netC, text_encoder,image_encoder, optimizerG, optimizerD, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC = netG.train(), netD.train(), netC.train()

    noise = torch.randn(batch_size, z_dim).to(device)

    mask = mask_correlated_samples(args)
    temperature = 0.5
    device = noise.get_device()
    criterion = NT_Xent(batch_size, temperature, mask, device)  # contrastive loss

    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        # prepare_data
        #加个分支
        imgs,imgs_2, sent_emb,sent_emb_2, words_embs,words_embs_2, keys ,sort_ind, sort_ind_2= prepare_data(data, text_encoder)
        imgs = imgs.to(device).requires_grad_()
        sent_emb = sent_emb.to(device).requires_grad_()
        words_embs = words_embs.to(device).requires_grad_()

        imgs_2 = imgs_2.to(device).requires_grad_()
        sent_emb_2 = sent_emb_2.to(device).requires_grad_()
        words_embs_2 = words_embs_2.to(device).requires_grad_()



        # predict real
        real_features = netD(imgs)
        pred_real, errD_real = predict_loss(netC, real_features, sent_emb, negtive=False)
        mis_features = torch.cat((real_features[1:], real_features[0:1]), dim=0)
        _, errD_mis = predict_loss(netC, mis_features, sent_emb, negtive=True)

        real_features_2 = netD(imgs_2)
        pred_real_2, errD_real_2 = predict_loss(netC, real_features_2, sent_emb_2, negtive=False)
        mis_features_2 = torch.cat((real_features_2[1:], real_features_2[0:1]), dim=0)

        #BERT-DFGAN-CL改造：errD_mis_2貌似是这次改造加的
        _, errD_mis_2 = predict_loss(netC, mis_features_2, sent_emb_2, negtive=True)

        # synthesize fake images

        # noise = torch.randn(batch_size, z_dim).to(device)
        fake = netG(noise, sent_emb)
        fake_features = netD(fake.detach())


        fake_2 = netG(noise,sent_emb_2)
        fake_features_2 = netD(fake_2.detach())


        _, errD_fake = predict_loss(netC, fake_features, sent_emb, negtive=True)
        _,errD_fake_2 = predict_loss(netC,fake_features_2,sent_emb_2,negtive=True)
        # MA-GP
        errD_MAGP = MA_GP(imgs, sent_emb, pred_real)
        errD_MAGP_2 = MA_GP(imgs_2, sent_emb_2, pred_real_2)
        # whole D loss
        errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
        errD_2 = errD_real_2 + (errD_fake_2 + errD_mis_2) / 2.0 + errD_MAGP_2

        errD += errD_2
        # update D
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()
        # update G
        fake_features = netD(fake)
        fake_features_2 = netD(fake_2)
        output = netC(fake_features, sent_emb)
        output_2 = netC(fake_features_2, sent_emb_2)
        # sim = MAP(image_encoder, fake, sent_emb).mean()
        errG = -output.mean()# - sim
        errG_2 = -output_2.mean()
        errG += errG_2

        #BERT-DFGAN-CL改造：contra_loss
        total_contra_loss = 0
        _, ori_indices = torch.sort(sort_ind, 0)
        _, ori_indices_2 = torch.sort(sort_ind_2, 0)
        _,cnn_code = image_encoder(fake)
        _,cnn_code_2 = image_encoder(fake_2)
        cnn_code = cnn_code[ori_indices]
        cnn_code_2 = cnn_code_2[ori_indices_2]
        cnn_code = l2norm(cnn_code, dim=1)
        cnn_code_2 = l2norm(cnn_code_2, dim=1)
        contrative_loss = criterion(cnn_code, cnn_code_2)
        total_contra_loss += contrative_loss * 0.2

        errG += total_contra_loss

        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()


def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp, truncation, trunc_rate, times):
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        #这些变量为什么注释呢？
        imgs,_, sent_emb,_, words_embs,_, keys,_,_= prepare_data(data, text_encoder)
        sent_emb = sent_emb.to(device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            if truncation==True:
                noise = truncated_noise(batch_size, z_dim, trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise,sent_emb)
        for j in range(batch_size):
            s_tmp = '%s/single/%s' % (save_dir, keys[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            ######################################################
            # (3) Save fake images
            ######################################################            
            if multi_gpus==True:
                filename = 'd%d_s%s.png' % (get_rank(),times)
            else:
                filename = 's%s.png' % (stamp)
            fullpath = '%s_%s.png' % (s_tmp, filename)
            im.save(fullpath)


def test(dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    fid = calculate_fid(dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch, \
                        times, z_dim, batch_size, truncation, trunc_rate)
    return fid


def calculate_fid(dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size()
    n_gpu = 1
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs,_, sent_emb,_, words_embs,_, keys,_,_ = prepare_data(data, text_encoder)
            sent_emb = sent_emb.to(device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                # output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                # pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                # pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluate Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def eval(dataloader, text_encoder, netG, device, m1, s1, save_imgs, save_dir,
                times, z_dim, batch_size, truncation=True, trunc_rate=0.86):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size()
    n_gpu = 1
    dl_length = dataloader.__len__()

    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs,_, sent_emb,_, words_embs,_, keys,_,_= prepare_data(data, text_encoder)
            sent_emb = sent_emb.to(device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb)
                if save_imgs==True:
                    save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                # output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                # pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                # pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            #print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png'%(time*dl_len*batch_size+batch_size*batch_n+j, get_rank())
        fullpath = osp.join(folder, filename)
        im.save(fullpath)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def sample_one_batch(noise, sent, netG, multi_gpus, epoch, img_save_dir, writer):
    fixed_results = generate_samples(noise, sent, netG)
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        if writer!=None:
            fixed_grid = make_grid(fixed_results.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results', fixed_grid, epoch)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)


def generate_samples(noise, caption, model):
    with torch.no_grad():
        fake = model(noise, caption)
    return fake


#########   MAGP   ########
def MA_GP(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def predict_loss(predictor, img_feature, text_feature, negtive):
    #代码去哪了？找不到predictor的实现
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.nn.ReLU()(1.0 - output).mean()
    else:
        err = torch.nn.ReLU()(1.0 + output).mean()
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err

#对应from miscc.losses import sent_loss, words_loss
def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # np.concatenate 拼接数组
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        # unsqueeze 升维，squeeze 降维
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    # torch.norm是对输入的Tensor求范数。dim=2代表从2维度计算范数，0列1行
    # ，2， 代表2范数，，它返回特征值向量：谱范数，即A’A矩阵的最大特征值的开平方
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    # bmm 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m）
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    # torch.clamp 压缩张量
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        # masked_fill_表示的意思是：在原tensor中，mask中对应元素为1的位置都用num填充。带_代表修改自己本身
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)

        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps
