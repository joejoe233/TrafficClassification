#加载训练集用于训练
import os
import random
import cv2 as cv
import numpy as np
import re
import copy
classkind= 10
labeldict = {}

def train_image(path,classname=None,labeldicts = None):
    global labeldict
    count = 0
    templabels = [0 for i in range(classkind)]
    images = []
    labels = []
    if labeldicts !=None:
        labeldict=labeldicts
    imagenamelist = []
    if classname==None:
        imagenamelist = [path+"\\"+name for name in os.listdir(path) if name.lower().endswith('jpg')] #生成一个列表 lower()把所有大写字母转换成小写字母
    else:
        imagenamelist = [path+"\\"+name for name in os.listdir(path) if name.lower().endswith('jpg')and name.lower().startswith(classname)]
    random.shuffle(imagenamelist)
    random.shuffle(imagenamelist) #随机排序
    for i in imagenamelist:
        image = cv.imread(i,flags=0) #读入图像
        image = image[:,:,np.newaxis] #添加数组维度
        images.append(image)
        pattern = re.compile('^[a-z]+')
        vpnpattern = re.compile('(vpn_[a-z]+)')
        name = i.split('\\')[-1]
        if name.startswith('vpn'):
            name = vpnpattern.findall(name.lower())[0]
        else:
            name = pattern.findall(name.lower())[0]
        if name in labeldict:
            label = labeldict[name]
            labels.append(label)
            count +=1
        else:
            labellength = len(labeldict)
            templabel = copy.deepcopy(templabels)
            templabel[labellength] = 1
            labeldict.update({name:templabel})
            label = templabel
            labels.append(label)
            count += 1
    images = np.array(images)
    labels = np.array(labels)
    if classname!=None:
        return images, labels
    else:
        return images,labels,labeldict
