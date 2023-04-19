from __future__ import print_function
#代码改动
from utils import mkdir_p
from utils import build_super_images
from modules import sent_loss, words_loss
from pretrain_DAMSM_config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_damsm_data
#代码改动
from models.DAMSM import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#代码改动：不加这个下面pycharm会灰色显示data
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from masks import mask_correlated_samples_2
from nt_xent import NT_Xent


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 50
#L2范数：L2范数是指向量各元素的平方和然后求平方根。
#L1范数可以进行特征选择，即让特征的系数变为0。
#L2范数可以防止过拟合，提升模型的泛化能力，有助于处理 condition number不好下的矩阵(数据变化很小矩阵求解后结果变化很大)
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='../cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir, criterion):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()

    #DAMSM主要有两个神经网络，文本编码器和图像编码器。其将句子的图像和单词的子区域映射到一个公共语义空间，从而在单词级别测量图像-文本相似度，以计算图像生成的细粒度损失。
    #文本编码器：采用双向长短期记忆网络（LSTM,RNN）
    #图像编码器：采用卷积神经网络（CNN），将图像映射到语义向量
    for step, data in enumerate(dataloader, 0):
        # print('step', step)

        rnn_model.zero_grad()
        cnn_model.zero_grad()

        # imgs, captions, cap_lens, \
        #     class_ids, keys = prepare_data(data)

        imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
        sort_ind, sort_ind_2 = prepare_damsm_data(data)


        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        words_features_2, sent_code_2 = cnn_model(imgs_2[-1])

        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # nef_2, att_sze_2 = words_features_2.size(1), words_features_2.size(2)


        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        #为什么会有1和2
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        words_emb_2, sent_emb_2 = rnn_model(captions_2, cap_lens_2, hidden)


        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        w2_loss0, w2_loss1, attn_maps_2 = words_loss(words_features_2, words_emb_2, labels,
                                                 cap_lens_2, class_ids_2, batch_size)
        w_total_loss0 += w2_loss0.data
        w_total_loss1 += w2_loss1.data
        loss += w2_loss0 + w2_loss1


        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        s2_loss0, s2_loss1 = \
            sent_loss(sent_code_2, sent_emb_2, labels, class_ids_2, batch_size)
        loss += s2_loss0 + s2_loss1
        s_total_loss0 += s2_loss0.data
        s_total_loss1 += s2_loss1.data

        _, ori_indices = torch.sort(sort_ind, 0)
        _, ori_indices_2 = torch.sort(sort_ind_2, 0)

        sent_emb = sent_emb[ori_indices]
        sent_emb_2 = sent_emb_2[ori_indices_2]

        # sent_emb = l2norm(sent_emb, dim=1)

        sent_emb = l2norm(sent_emb, dim=1)
        sent_emb_2 = l2norm(sent_emb_2, dim=1)

        contrative_loss = criterion(sent_emb, sent_emb_2)
        loss += contrative_loss



        #

        # mse_loss = nn.MSELoss(reduction='sum')
        # q_out = netD.Q_NET(fake_features)
        # l2_loss = mse_loss(sent_code, sent_emb)
        # batch_size = region_features.size(0)
        # l2_loss = l2_loss / batch_size
        # l2_loss = l2_loss * 0.1
        # print(l2_loss)

        # loss += l2_loss

        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            # img_set, _ = \
            #     build_super_images(imgs[-1].cpu(), captions,
            #                        ixtoword, attn_maps, att_sze)
            # if img_set is not None:
            #     im = Image.fromarray(img_set)
            #     fullpath = '%s/attention_maps%d.png' % (image_dir, step)
            #     im.save(fullpath)
    return count

#评价相关性的重要指标
def evaluate(dataloader, cnn_model, rnn_model, batch_size, criterion):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        # real_imgs, captions, cap_lens, \
        #         class_ids, keys = prepare_data(data)

        real_imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
        sort_ind, sort_ind_2 = prepare_damsm_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    # 这是什么？imgSize？
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    #为什么这里有一个assert？怎么触发？
    assert dataset



    #数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。在训练模型时使用到此函数，用来 把训练数据分成多个小组 ，此函数 每次抛出一组数据 。直至把所有的数据都抛出。就是做一个数据的初始化。
    #使用enumerate函数对数据集切分
    #pickle n.腌渍、菜酱
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    mask = mask_correlated_samples_2(batch_size)

    temperature = 0.5
    device = labels.get_device()
    criterion = NT_Xent(batch_size, temperature, mask, device)


    try:
        #关键代码
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            #adam优化器
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir, criterion)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size, criterion)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

