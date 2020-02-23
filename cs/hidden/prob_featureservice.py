#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
# sys.path.insert(0,"/home/swf/work/caffe/python/")
import matplotlib.pyplot as plt
import caffe
import os
import scipy.io
import sklearn
import RSA_EnDe
import skimage
# from sklearn.externals import joblib
# Make sure that caffe is on the python path:

caffe.set_device(0)
caffe.set_mode_gpu()
# def GetFileList(dir, fileList):
#     newDir = dir
#     if os.path.isfile(dir):
#         fileList.append(dir.decode('gbk'))
#     elif os.path.isdir(dir):
#         for s in os.listdir(dir):
#             #如果需要忽略某些文件夹，使用以下代码
#             if s.endswith(".txt") or s.endswith(".sh") or s.endswith(".py"):
#                 continue
#             #if int(s)>998 and int(s) < 1000:
#             newDir=os.path.join(dir,s)
#             GetFileList(newDir, fileList)
#     return fileList


def load_model(path):
    caffe_root = path  # this file is expected to be in {caffe_root}/examples
    #sys.path.insert(0, caffe_root + 'python')
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'
    #  model = 'models/casiaface/casia.caffemodel'

    # model = '0601.caffemodel'
    model = 'prob_0408.caffemodel'

    #model = 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
    #model = 'models/casiaface/alignmixdata_iter_400000.caffemodel'
    #model = 'models/generateface/order_iter_330000.caffemodel'
    #model = 'models/casiaface/deepid'

    # deploy = 'deploy2.prototxt'
    deploy = 'prob_deploy.prototxt'

    try:
        RSA_EnDe.Descrypt(caffe_root ,deploy)
        RSA_EnDe.Descrypt(caffe_root , model)
    except Exception, e:
        print(e)
        print("Has been Descrypted")
    if not os.path.isfile(caffe_root + model):
        print(caffe_root + model)
    caffe.set_mode_gpu()
    net = caffe.Net(caffe_root + deploy,
            caffe_root + model,
            caffe.TEST)
    try:
        RSA_EnDe.Encrypt(caffe_root,deploy)
        RSA_EnDe.Encrypt(caffe_root , model)
    except Exception, e:
        print(e)
        print("Has been Encrypted")
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'newmean.npy').mean(1).mean(1))   # mean pixel.mean(1)
    # transformer.set_mean('data', np.load(r"E:\caffe-master\python\caffe\imagenet\ilsvrc_2012_mean.npy").mean(1).mean(1))   # mean pixel.mean(1)
    # print np.load(r"E:\caffe-master\python\caffe\imagenet\ilsvrc_2012_mean.npy").shape
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(10, 3, 227, 227)
    return net, transformer



def pred(image1, net, transformer):
    caffe.set_mode_gpu()
    mat = []
    nn = 0
    #for image in [image1, image2]:
    try:
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image1))
    except Exception, e:
        print nn
        print str(e)
        nn += 1
        #continue
    #  out = net.forward()
    #  print("Predicted class is #{}.".format(out['prob'].argmax()))
    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    net.forward()  # call once for allocation
    prob = net.blobs['prob'].data[1][0] # net.blobs['prob'].data[0]
    print prob.shape
    return prob




