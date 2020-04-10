#对训练好的模型进行评判指标计算
import tensorflow as tf
from LoadImage import train_image
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import copy


images = []
#模型的存放位置
modelpath = "E:\\Flow classification\\TransformImage\\model(10)"
#测试集存放位置
testimagepath = "E:\\Flow classification\\TransformImage\\test\\"
#模型的名字
modelname = 'model.ckpt-100.meta'
#实验的次数
exp_count = 44
#分类的大小
kindclass=10
#分类和标签的对应关系
resultpath = 'E:\\Flow classification\\TransformImage\\Result\\result{}.txt'.format(exp_count)

#ROC曲线是一种显示在分类模型所在分类阈值下的效果的图表
def ROC(classdict,testlabel,pred):
    fpr = dict()
    tpr = dict()
    thresholds =dict()
    roc_auc = dict()
    for i in range(len(classdict)):
        t = testlabel[:, i]
        p = pred[:, i]
        sum1 = 0
        for j in range(len(t)):
            sum1 =sum1+(t[j]-p[j])
        #调用roc_curve计算每中阈值的fpr和tpr值
        fpr[i],tpr[i],thresholds[i] = roc_curve(testlabel[:,i].ravel(),pred[:,i].ravel())
        roc_auc[i] = auc(fpr[i],tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classdict))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classdict)):
        mean_tpr +=interp(all_fpr,fpr[i],tpr[i])
    mean_tpr/=len(classdict)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"] ,tpr["macro"] )

    #画图线的粗细
    lw = 2
    #画各种曲线的平均值曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    #定义曲线颜色
    colors = []
    if kindclass==20:
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'crimson', 'orchid', 'purple', 'indigo', 'black', 'slategray','blue','darkslateblue','yellow','red','cyan','orange','tan','brown','olive','gold'])
    else:
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','pink','crimson','orchid','purple','indigo','black','slategray'])
    #对每个分类画曲线
    for i, color in zip(range(len(classdict)), colors):
        name = ''
        for kv in classdict.items():
            if kv[1] == i:
                name = kv[0]
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(name, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    font = {'size':7}
    plt.legend(loc="lower right",prop=font)
    plt.show()



with tf.Session().as_default() as sess:
    classifidicts = {}
    classdict = {}
    labelexa = [0for i in range(kindclass)]
    with open(resultpath,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            labeltemp = line.split(' ')
            classname = labeltemp[0]
            classflagnum = int(labeltemp[-1])
            classflaglist = copy.deepcopy(labelexa)
            classflaglist[classflagnum] = 1
            classifidicts.update({classname: classflaglist})
            classdict.update({classname:classflagnum})
    #调用模型
    saver = tf.train.import_meta_graph(modelpath+'\\'+modelname)
    saver.restore(sess,tf.train.latest_checkpoint(modelpath))
    #获取输出
    pred= tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()
    x1 = graph.get_operation_by_name('input/x').outputs[0]
    keep_prob1 = graph.get_operation_by_name('input/keep_prob').outputs[0]
    test_x, test_y ,_= train_image(testimagepath,labeldicts=classifidicts)
    #获取输出
    predict = sess.run(pred, feed_dict={x1: test_x, keep_prob1: 1.0})
    #画根据结果画ROC曲线
    ROC(classdict, test_y, predict)
