#测试模型或预测结果
import tensorflow as tf
import cv2 as cv
import numpy as np
import copy
images = []
#模型存放位置
modelpath = "E:\\Flow classification\\TransformImage\\model(10)"
#测试集文件位置
testimagepath = "E:\\Flow classification\\TransformImage\\test"
#模型名字
modelname = 'model.ckpt-100.meta'
#用于测试的图片名
testimagename = 'facebook_test279.jpg'
#种类
kindclass = 10
#实验次数
exp_count = 44
#结果存放位置
resultpath = 'E:\\Flow classification\\TransformImage\\Result\\result{}.txt'.format(exp_count)
with tf.Session().as_default() as sess:
    labeldicts = {}
    labelexa = [0 for i in range(kindclass)]
    #获取标签
    with open(resultpath,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            labeltemp = line.split(' ')
            classname = labeltemp[0]
            classflagnum = int(labeltemp[-1])
            classflaglist = copy.deepcopy(labelexa)
            classflaglist[classflagnum] = 1
            labeldicts.update({classname:classflagnum})
    #使用模型
    saver = tf.train.import_meta_graph(modelpath+"\\"+modelname)
    saver.restore(sess,tf.train.latest_checkpoint(modelpath))
    image = cv.imread(testimagepath+'\\'+testimagename,flags=0)
    image = image[np.newaxis,:,:,np.newaxis]
    pred= tf.get_collection('pred_network')[0]
    prednum = tf.arg_max(pred,1)
    graph = tf.get_default_graph()
    x1 = graph.get_operation_by_name('input/x').outputs[0]
    y1 = graph.get_operation_by_name('input/y').outputs[0]
    keep_prob1 = graph.get_operation_by_name('input/keep_prob').outputs[0]
    result1 = sess.run(prednum, feed_dict={x1: image, keep_prob1: 1.0})
    for onelabel in labeldicts.items():
        if onelabel[1]== result1:
            print("预测结果为",onelabel[0])