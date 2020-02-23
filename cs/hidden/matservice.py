import SocketServer
from SocketServer import StreamRequestHandler as SRH
from time import ctime
from time import sleep
import numpy as np
import featureservice
from PIL import Image
import cv2
import sys,getopt

import prob_featureservice
import argparse

HEIGHT=227
WIDTH=227
IMG_SIZE=HEIGHT*WIDTH*3
host = '127.0.0.1'
port = 3202
addr = (host, port)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, default='/home1/FileDir/ssd/caffe/cs/RSA/', help='model path')
args = parser.parse_args()

net, transformer = featureservice.load_model(args.model_path)
prob_net, prob_transformer = prob_featureservice.load_model(args.model_path)
#net,transformer=featureservice.load_models("models/scene/idcard/deepid_flip_train_iter_20000.caffemodel",
#"models/scene/idcard/deploy.prototxt","models/scene/idcard/idface.npy",[256,3,55,55])
#net,transformer=featureservice.load_models("models/mfm/model/DeepFace_set003_net_iter.caffemodel",
#"models/mfm/proto/DeepFace_set003.prototxt","models/mfm/data/alignCASIA-WebFace_mean_2.npy",[20,1,128,128])
RECV_SIZE = 1024

class Servers(SRH):
    def recv_img(self):
        i=0
        rev_data=''
        while i<IMG_SIZE:
            rev_data+=self.request.recv(IMG_SIZE-i)
            i=len(rev_data)
        data=np.fromstring(rev_data,dtype='uint8')
        data=data.reshape((HEIGHT,WIDTH,3))
        #new_im = Image.fromarray(data)
        #new_im.show()
        #new_im.save('new.jpg')
        #print data.shape

        return data

    def handle(self):
        print 'Got connection from ', self.client_address
        # self.wfile.write('Connection %s:%s at %s succeed!' % (host, port, ctime()))
        # while True:
        try:
            rev_data = self.recv_img()
            cv2.imwrite('save.jpg',rev_data)
        # self.wfile.write('Recv: %s' % rev_data)
        except Exception as e:
            print str(e)
        #dirs = rev_data.decode('utf-8')
        sim = featureservice.pred('save.jpg', net, transformer,prob_net, prob_transformer)
        self.request.sendall(str(sim))
if __name__ == "__main__":
    print 'Server is running....'
    server = SocketServer.ThreadingTCPServer(addr,Servers)
    server.serve_forever()
