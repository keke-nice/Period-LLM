# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
import torch
import MyDataset
import MyLoss
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
from datetime import datetime
import os
import time
from utils import Logger, time_to_str
from timeit import default_timer as timer
import time
import random
import wandb




if __name__ == '__main__':


    args = utils.get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    GPU = args.GPU
    batch_size_num = args.batchsize
    num_workers = args.num_workers
    epoch_num = args.epochs
    learning_rate = args.lr
    wandb.login()
    entity = "rppg"
    settings = wandb.Settings(job_name=args.job_name)
    run = wandb.init(
    # Set the project where this run will be logged
    entity = entity,
    project= args.project_name,
    name=args.job_name,
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
    },
    )
    print(args)


    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    log = Logger()
    log.open('./Result_log/' + 'text_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')

    # 数据集
  
    dataset = MyDataset.Data_countix(json_path=args.json_path, input_type=args.input_type, image_size=args.image_size, frame_num=args.frame_num)
    data_loader = DataLoader(dataset, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    
    BaseNet = model.BaseNet(args, my_type=args.my_type, input_type=args.input_type, device=device, visual=args.visual, image_size=args.image_size)
    if args.ckpts:
        checkpoint = torch.load(args.ckpts, map_location='cpu')
        state_dic = checkpoint.state_dict()
        msg = BaseNet.load_state_dict(state_dic,strict=False)
        print(msg)
        print('load BaseNet checkpoint from %s'%args.ckpts) 

    BaseNet.to(device=device)
    optimizer_rPPG = torch.optim.Adam(BaseNet.parameters(), lr=learning_rate)
   
    scaler = torch.cuda.amp.GradScaler()
    if args.my_type == 'mix':
        data_iter_video = data_loader_video.__iter__()
        data_iter_per_epoch_0 = len(data_iter_video)
        data_iter_text = data_loader_text.__iter__()
        data_iter_per_epoch_1 = len(data_iter_text)
    else:
        data_iter = data_loader.__iter__()
        data_iter_per_epoch_0 = len(data_iter)

    max_iter = args.max_iter
    start = timer()

    for iter_num in range(max_iter + 1):
        BaseNet.train()
        if args.my_type == 'mix':
            if (iter_num % data_iter_per_epoch_0 == 0):
                data_iter_video = data_loader_video.__iter__()
            if (iter_num % data_iter_per_epoch_1 == 0):
                data_iter_text = data_loader_text.__iter__()
            probability = random.random()
            # print(probability)
            if probability < args.P_video: 
                images, Question_video, Answer_video = data_iter_video.__next__()
                images = images.float().to(device=device)
                optimizer_rPPG.zero_grad()
                BaseNet.input_type = 'video'
                # print(Question_video)
                probability1 = random.random()
                if probability1 < 0.5:
                  QA_loss = BaseNet(images, Question_video[0], Answer_video[0])
                else:
                  QA_loss = BaseNet(images, Question_video[1], Answer_video[1])
            else:
                Question_text, Answer_text = data_iter_text.__next__() 
                optimizer_rPPG.zero_grad()
                BaseNet.input_type = 'text'
                QA_loss = BaseNet(None, Question_text, Answer_text)
        else:
            if (iter_num % data_iter_per_epoch_0 == 0):
                data_iter = data_loader.__iter__()
            if args.input_type=='video':
                images, Question, Answer = data_iter.__next__()
                images = images.float().to(device=device)
                optimizer_rPPG.zero_grad()
                BaseNet.k = 0.05 * (iter_num / max_iter)
                 # 更新 AdjustGradHook 中的 k 值
                if hasattr(BaseNet, 'adjust_grad_hook'):
                    BaseNet.adjust_grad_hook.update_k(BaseNet.k)
                QA_loss = BaseNet(images, Question, Answer)
            elif args.input_type=='text':
                Question, Answer = data_iter.__next__() 
                optimizer_rPPG.zero_grad()
                QA_loss = BaseNet(x=None, question_all=Question, caption_GT=Answer)
            elif args.input_type=='double':
                images, Question, Answer_GT, Answer_pre = data_iter.__next__() 
                images = images.float().to(device=device)
                optimizer_rPPG.zero_grad()
                QA_loss, c_loss, b_loss, anti_loss = BaseNet(x=images, question_all=Question, caption_GT=Answer_GT, caption_pre=Answer_pre)

        k = 1.0 + 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0

        loss = k * QA_loss#\
               #+ (src_loss_aug_0 + src_loss_aug_1 + src_loss_aug_2 + src_loss_aug_3)
        if torch.sum(torch.isnan(loss)) > 0:
            print('Nan')
            break
        else:
            loss.backward()
            optimizer_rPPG.step()
            
        if iter_num % 500 == 0:
            # 清空未使用的缓存
            torch.cuda.empty_cache()
            # torch.save(BaseNet, './result_Model_new/' + args.job_name +str(iter_num)+ '.pth')
            # print('saveModel As ' + args.job_name + '.pth')
            log.write(
                'Train Inter:' + str(iter_num)\
                + ' | loss:  ' + str(loss.data.cpu().numpy()) \
                + ' |' + time_to_str(timer() - start, 'min'))
            log.write('\n')
            wandb.log({"loss": loss})
            wandb.log({"QA_loss": QA_loss})
            # wandb.log({"pos_loss": c_loss})
            # wandb.log({"neg_loss": b_loss})
            # wandb.log({"anti_loss": anti_loss})

      
        if (iter_num >= 5000) and (iter_num % 100000 == 0):
              if not os.path.exists('./result_Model_new'):
                os.makedirs('./result_Model_new')
              torch.save(BaseNet, './result_Model_new/' + args.job_name +str(iter_num)+ '.pth')
              print('saveModel As ' + args.job_name + '.pth')
