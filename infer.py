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
import json
import pdb

if __name__ == '__main__':

    args = utils.get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
    pre_train_model = args.infer_model


    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index
    save_path = './Result/' + args.job_name
    print(args)



    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 数据集
    dataset = MyDataset.Data_countix(json_path=args.json_path, input_type=args.input_type, image_size=args.image_size, frame_num=args.frame_num)
    tgt_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    BaseNet = model.BaseNet(args=args, my_type=args.my_type, input_type=args.input_type, device=device, visual=args.visual, image_size=args.image_size)
    BaseNet = torch.load(pre_train_model, map_location=device)
    BaseNet.to(device=device)
    BaseNet.input_type = args.input_type
    

    start = timer()

    BaseNet.eval()
    Question_list = []
    Caption_list = []
    Caption_pre_list = []
    video_path_list = []
    data_json = []
    if args.input_type=='video':
      for step, (images,   Question, Answer) in enumerate(tgt_loader):
          data = Variable(images).float().to(device=device)
          caption_pre = BaseNet.Inference(data, Question)
          # video_path_list.append(video_path)
          Question_list.append(Question)
          Caption_list.append(Answer)
          Caption_pre_list.append(caption_pre)
          print('caption', Answer)
          print('caption_pre', caption_pre)
          # temp_data = {
          #                   "video": video_path[0],
          #                   "QA": 
          #                       {
          #                           "q": Question[0],
          #                           "a_GT": Answer[0],
          #                           "a_pre": caption_pre[0],
          #                       }
          #               }
          # data_json.append(temp_data)
          # if step==1:
          #     break
    elif args.input_type=='text':
      for step, (Question, Answer) in enumerate(tgt_loader):
          
          # pdb.set_trace()

          caption_pre = BaseNet.Inference(x=None, question_all=Question)
          Question_list.append(Question)
          Caption_list.append(Answer)
          Caption_pre_list.append(caption_pre)
          print('caption', Answer)
          print('caption_pre', caption_pre)
          

        

    # 保存list到文件
    # with open(args.saveQ_path, 'w') as f:
    #     json.dump(Question_list, f)
    with open('./answer_GT.json', 'w') as f:
        json.dump(Caption_list, f)
    with open('./answer_pre.json', 'w') as f:
        json.dump(Caption_pre_list, f)
    with open('./QA.json', 'w') as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)
    



