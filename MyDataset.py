# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
from numpy.fft import fft, ifft, rfft, irfft
from torch.autograd import Variable
import random
import json
from decord import VideoReader
from decord import cpu
import torch

class Data_countix(Dataset):
    def __init__(self, json_path, input_type, frame_num, image_size):
        self.input_type = input_type
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file) 
        self.num = len(self.data) 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])
        
        self.frame_num = frame_num

    def __len__(self):
        return self.num

    def loadvideo_decord_origin(self, video_path, frame_num, clip_len=8, frame_sample_rate=2,num_segment=1):
        fname = video_path
        vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        # handle temporal segments
        converted_len = int(clip_len * frame_sample_rate) #16
        seg_len = len(vr) //num_segment #74
        # duration = max(len(vr) // vr.get_avg_fps(), frame_num)#vr.get_avg_fps()=视频帧率
        duration = max(5*len(vr) // vr.get_avg_fps(),20)#vr.get_avg_fps()=视频帧率

        all_index = []
        for i in range(num_segment):
            index = np.linspace(0, seg_len, num=int(duration))
            index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index)) #从视频中等距离取8帧
        
        if len(all_index) <= 100:
            all_index = all_index
        # elif len(all_index) < 100:
        #     all_index = all_index[::int(2)]
        elif len(all_index) < 200:
            all_index = all_index[::int(4)]
        elif len(all_index) < 400:
            all_index = all_index[::int(6)]
        elif len(all_index) < 800:
            all_index = all_index[::int(8)]
        elif len(all_index) < 1000:
            all_index = all_index[::int(10)]
        elif len(all_index) < 2000:
            all_index = all_index[::int(20)]
        elif len(all_index) < 4000:
            all_index = all_index[::int(40)]
        elif len(all_index) < 6000:
            all_index = all_index[::int(60)]
        elif len(all_index) < 8000:
            all_index = all_index[::int(80)]
        elif len(all_index) < 10000:
            all_index = all_index[::int(100)]
        else:
            all_index = all_index[::int(200)]
            
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()#[9,360,640,3] 一个视频每秒取1帧，最小8帧
        return buffer

    def __getitem__(self, idx):
        idx = idx
        if self.input_type == 'video':
            Question_list = []
            Answer_list = [] 

            video_path = self.data[idx]['video']
            Question = self.data[idx]['QA']['q']
            Answer = self.data[idx]['QA']['a']
       
            video_frames = self.loadvideo_decord_origin(video_path, self.frame_num) #[8,360,640,3]
            tmp = []
            for i, img in enumerate(video_frames):
                tmp.append(self.transform(img).unsqueeze(0)) #[1,3,384,384]
            images = torch.cat(tmp)#[9,3,384,384]
            return images, Question, Answer
        elif  self.input_type == 'text':  
            # key_idx = 'text_' + str(idx+1)
            # text = self.data[key_idx]['text']
            # Question = 'Count the number of repeated characters in this section and the number of times they are repeated: ' + text
            # duplicates = self.data[key_idx]['duplicates']
            # Answer = 'Here are the repeating characters in this text and their number: ' + str(duplicates) 
            Question = self.data[idx]['QA']['q']
            Answer = self.data[idx]['QA']['a']
            return Question, Answer
      
        elif self.input_type == 'double':
            video_path = self.data[idx]['video']
            # print(video_path)
            Question = self.data[idx]['QA']['q']
            Answer_GT = self.data[idx]['QA']['a_GT']
            Answer_pre = self.data[idx]['QA']['a_pre']
            video_frames = self.loadvideo_decord_origin(video_path, self.frame_num) #[8,360,640,3]
            tmp = []
            for i, img in enumerate(video_frames):
                tmp.append(self.transform(img).unsqueeze(0)) #[1,3,384,384]
            images = torch.cat(tmp)#[9,3,384,384]
            return images, Question, Answer_GT, Answer_pre
        

        



