# -*- coding: utf-8 -*-

import os
from os import listdir
from os.path import isfile, join, isdir
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

image_types = (".mp4", ".avi",".tif",".jpg")
image_dir = "Testing_Gen_image/"

#path to test vidoes
# video_dir = "UCSD_Anomaly_Dataset/UCSDped2/Test/Test005/"
# video_dir = "Avenue1/testing_videos/"
# video_dir = "Avenue Dataset/testing_videos/"
video_dir = "ShanghaiTech/testing/frames/01_0051/"


def list_videos(basePath, contains=None):

    return list_files(basePath, validExts=image_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # Fetch files present in filenames
        for filename in filenames:
           
            if contains is not None and filename.find(contains) == -1:
                continue

            # getting file extension
            ext = filename[filename.rfind("."):].lower()
  
            # check if the file has valid extension
            if validExts is None or ext.endswith(validExts):
                # build the path to the frame/video and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

videoPaths = list(list_videos(video_dir))
# print(len(videoPaths))
all_frames = []


for path in videoPaths:
    cap = cv2.VideoCapture(path)
    w = 128
    h = 128
    fc = 0
    ret = True

    while True:
        ret, frame = cap.read()
        if ret == True:
            resized_frame = cv2.resize(frame, (128, 128))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            all_frames.append(np.array(resized_frame,dtype=np.float64) / 255.0)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# print(len(all_frames))

all_frames = np.array(all_frames)
# print(all_frames.shape)
size = all_frames.shape[0] - 7
sequences = np.zeros((size, 7, 128, 128, 3))

for i in range(0, size):
  clip = np.zeros((7, 128, 128, 3))
  for j in range(0, 7):
    clip[j] = all_frames[i + j, :, :, :]
  sequences[i] = clip

print("sequence",sequences.shape)

padding=np.zeros((7,7,128,128,3))

for i in range(0,7):

    im=np.zeros((7,128,128,3))
    for j in range(0,7):
        im[j]=all_frames[sequences.shape[0]+i,:,:,:]
    padding[i]=im

test_data=np.concatenate((sequences, padding), axis=0)


print("test data shape",test_data.shape)

#loading the trained  GAN model
# g=load_model('Model_Save/Model_Save_UCSD_PED2/g5000.h5',compile=False)
# g=load_model('Model_Save/Model_Save_UCSD_PED1/g5000.h5',compile=False)
# g=load_model('Model_Save/Model_Save_Avenue/g5000.h5',compile=False)
g=load_model('Model_Save/Model_Save_ShanghaiTech/g5000.h5',compile=False)

# print(g)

# testing the model
gan_x=g.predict(test_data)

# Calculating the reconstruction cost for each frames in test video 
reconstruction_cost = np.array([np.linalg.norm(np.subtract(test_data[i],gan_x[i])) for i in range(0,test_data.shape[0])])

# Path to save the results
folder_name="Generated_Anomaly_Score_ShanghaiTech/Test01/"
# folder_name="Generated_Anomaly_Score_UCSDPed2/"
# folder_name="Generated_Anomaly_Score_UCSDPed1/"
# folder_name="Generated_Anomaly_Score_Avenue/"

# Calculating the Anomaly Score for each frame in test video
anomaly_score = (reconstruction_cost - np.min(reconstruction_cost)) / np.max(reconstruction_cost)

# Calculating the Regularity Score for each frame in test video
regularity_score = 1.0 - anomaly_score

# print(regularity_score)
score=regularity_score.tolist()

da={'score':score}
df=pd.DataFrame(da)
df.to_csv(folder_name + "/result051.csv")

plt.plot(regularity_score)
plt.title("Test Video 51")
plt.ylabel('regularity score')
plt.xlabel('Frames')
plt.savefig("ShanghaiTech_Regularity_Score/051")