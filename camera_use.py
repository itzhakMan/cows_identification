# %% [markdown]
# # One Shot Learning with Siamese Networks
#
# This is the jupyter notebook that accompanies

# %% [markdown]
# ## Imports
# All the imports are defined here
#matplotlib
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2
from classic_cnn import *
import time
import threading
import FaceMatch_threadClass

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0

    net = SiameseNetwork()
    net.load_state_dict(torch.load("./trained_siam_net_model.pt"))
    net.eval()

    thread_face_match = FaceMatch_threadClass.CustomThread(net, "./target")
    last_prediction = None

    while True:
        # Read the frame
        success, img = cap.read()
        # Display


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img=img, text=str(int(fps)), org=(10, 70),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)


        if not thread_face_match.isAlive():
            last_prediction = thread_face_match.value
            cv2.putText(img=img, text=last_prediction, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 255, 0), thickness=3)
            thread_face_match = FaceMatch_threadClass.CustomThread(net, "./target")
            cv2.imwrite('./target/test/current.jpeg', img)
            #result = face_match(net, "./target")
            thread_face_match.start()
        else:
            cv2.putText(img=img, text=last_prediction, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(0, 255, 0), thickness=3)


        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break



    # Release the VideoCapture object
    cap.release()

    # ## Training Time!
    #train_model()
    # net = SiameseNetwork()
    # net.load_state_dict(torch.load("./trained_siam_net_model.pt"))
    # net.eval()
    #
    #
    # # ## Some simple testing
    # # The last 3 subjects were held out from the training, and will be used to test. The Distance between each image pair denotes the degree of similarity the model found between the two images. Less means it found more similar, while higher values indicate it found them to be dissimilar.
    #
    # result = face_match(net, "./target")
    # #visualise_differences(net)



if __name__ == '__main__':
    main()