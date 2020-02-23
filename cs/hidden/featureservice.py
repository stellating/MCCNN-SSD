#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
import os
# change this to your caffe root dir
#caffe_root = '/home1/FileDir/ssd/caffe'
sys.path.insert(0, os.path.join(os.getcwd(),'python'))
print(os.path.join(os.getcwd(),'python'))

import matplotlib.pyplot as plt
import caffe


import scipy.io
import sklearn
#RSA
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP
import os
import base64
import cv2
import prob_featureservice
import time
import copy

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

#RSA
def Encrypt(path,name):    
    filename = os.path.join(path,name)   
    data = ''
    with open(filename, 'rb') as f:
        data = f.read()
    with open(filename, 'wb') as out_file:
        # 收件人秘钥 - 公钥
        recipient_key = RSA.import_key(open(os.path.join(path,'my_rsa_public.pem')).read())
        session_key = get_random_bytes(16)
        # Encrypt the session key with the public RSA key
        cipher_rsa = PKCS1_OAEP.new(recipient_key)
        out_file.write(cipher_rsa.encrypt(session_key))
        # Encrypt the data with the AES session key
        cipher_aes = AES.new(session_key, AES.MODE_EAX)
        ciphertext, tag = cipher_aes.encrypt_and_digest(data)
        out_file.write(cipher_aes.nonce)
        out_file.write(tag)
        out_file.write(ciphertext)
        
def Descrypt(path,name):
    filename = os.path.join(path,name)
    code = 'nooneknows'
    with open(filename, 'rb') as fobj:
        private_key = RSA.import_key(open(os.path.join(path,'my_private_rsa_key.bin')).read(), passphrase=code)
        enc_session_key, nonce, tag, ciphertext = [ fobj.read(x) 
                                                    for x in (private_key.size_in_bytes(), 
                                                    16, 16, -1) ]
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
        cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
        data = cipher_aes.decrypt_and_verify(ciphertext, tag)
    
    with open(filename, 'wb') as wobj:
        wobj.write(data) 

def load_model(path):
    caffe_root = path  # this file is expected to be in {caffe_root}/examples
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'
    #  model = 'models/casiaface/casia.caffemodel'
    model = 'detection_0408.caffemodel'
    #model = 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
    #model = 'models/casiaface/alignmixdata_iter_400000.caffemodel'
    #model = 'models/generateface/order_iter_330000.caffemodel'
    #model = 'models/casiaface/deepid'

    try:
        Descrypt(caffe_root,'deploy.prototxt')
        Descrypt(caffe_root,model)
    except Exception, e:
        print("Has been Descrypted")
    if not os.path.isfile(caffe_root + model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_gpu()
    net = caffe.Net(caffe_root + 'deploy.prototxt',
            caffe_root + model,
            caffe.TEST)
    try:
        Encrypt(caffe_root, 'deploy.prototxt')
        Encrypt(caffe_root,model)
    except Exception, e:
        print("Has been Encrypted")
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    image_resize_width = 300
    image_resize_height = 300
    net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
    #net.blobs['data'].reshape(10, 3, 227, 227)
    return net, transformer



def pred(image1, net, transformer,prob_net, prob_transformer):
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
    #net.forward()  # call once for allocation

    #print net.blobs['mbox_conf_softmax'].shape
    #prob = net.blobs['mbox_conf_softmax'].data[0] # net.blobs['prob'].data[0]
    #print prob.shape
    #return prob

    detections = net.forward()['detection_out']
    #print detections
    #det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

# Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]
    top_conf = det_conf[top_indices]
    #top_label_indices = det_label[top_indices].tolist()
    #top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    if not os.path.exists('cropimg'):
        os.mkdir('cropimg')
    if not os.path.exists('resultimg'):
        os.mkdir('resultimg')
    result_name = str(int(time.time()*10))
    try:
        image_mat = cv2.imread(image1, cv2.IMREAD_COLOR)
    except Exception as e:
	print e
	traceback.print_exec()
    image_h = image_mat.shape[0]
    image_w = image_mat.shape[1]
    
    prob = []
    top_xmin_loc = []
    top_ymin_loc = []
    top_xmax_loc = []
    top_ymax_loc = []
    image_result = copy.copy(image_mat)
    if len(top_indices) > 0:
	for i in range(len(top_indices)):
    	    xmin_loc = top_xmin[i]*image_w
    	    ymin_loc = top_ymin[i]*image_h
    	    xmax_loc = top_xmax[i]*image_w
    	    ymax_loc = top_ymax[i]*image_h
    	    crop_img = image_mat[int(round(ymin_loc)):int(round(ymax_loc)),int(round(xmin_loc)):int(round(xmax_loc))]
    	    cv2.imwrite('cropimg/' + str(i)+'.jpg',crop_img)
    	    cv2.rectangle(image_result,(int(round(xmin_loc)),int(round(ymin_loc))),(int(round(xmax_loc)),int(round(ymax_loc))),(55,255,155),5)

    	    prob.append(prob_featureservice.pred('cropimg/' + str(i)+'.jpg', prob_net, prob_transformer))
    	    top_xmin_loc.append(xmin_loc)
    	    top_ymin_loc.append(ymin_loc)
    	    top_xmax_loc.append(xmax_loc)
    	    top_ymax_loc.append(ymax_loc)

    else:
	cv2.imwrite('cropimg/0.jpg',image_mat)
	prob.append(prob_featureservice.pred('cropimg/0.jpg', prob_net, prob_transformer))
	top_xmin_loc.append(0)
	top_ymin_loc.append(0)
	top_xmax_loc.append(image_w)
	top_ymax_loc.append(image_h)
    cv2.imwrite('resultimg/' + result_name +'.jpg',image_result)
    return [top_xmin_loc,top_ymin_loc,top_xmax_loc,top_ymax_loc,prob]

    





