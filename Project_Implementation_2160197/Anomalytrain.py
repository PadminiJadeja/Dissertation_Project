# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from keras.layers import TimeDistributed, Conv2D, LeakyReLU, BatchNormalization, Dense, ConvLSTM2D, Conv2DTranspose, Input, GlobalAveragePooling2D,SeparableConv2D
import keras
from keras.models import Model
import keras.backend as K
from keras.activations import tanh
from keras import backend
from keras.constraints import Constraint
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
import numpy as np
from imutils import paths
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
import gc
import tensorflow as tf

# path to training videos
# video_dir = "UCSD_Anomaly_Dataset/UCSDped1/Train/"
video_dir = "ShanghaiTech/training/videos/"
# video_dir = "UCSD_Anomaly_Dataset/UCSDped2/Train/"
# video_dir = "Avenue Dataset/training_videos/"

# path to save reconstruucted frames during training
image_dir = "Generated_Image_ShanghaiTech/"
# image_dir = "Generated_Image_UCSDped1/"
# image_dir = "Generated_Image_UCSDped2/"
# image_dir = "Generated_Image_Avenue/"

# path to save model
# model_dir = "Model_Save/Model_Save_UCSD_PED1/"
# model_dir = "Model_Save/Model_Save_UCSD_PED2/"
model_dir = "Model_Save/Model_Save_ShanghaiTech/"
# model_dir = "Model_Save/Model_Save_Avenue/"

image_types = (".mp4", ".avi",".tif",".jpg")

def list_videos(basePath, contains=None):

    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):

      for (rootDir, dirNames, filenames) in os.walk(basePath):
            print(rootDir,dirNames,filenames)

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

# list of all training video paths
videoPaths = list(list_videos(video_dir))
print(len(videoPaths))
all_frames = []

# function to sort the files in ascending order based on file names
def function_work(x):
    y = x.rsplit('.', 2)[-2]
    return ('log' not in x, int(y) if y.isdigit() else float('inf'), x)

videoPaths = sorted(videoPaths, key=function_work, reverse=False)

# Resizing the each video frames to (128 X 128) and normalizing the pixel values
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
            all_frames.append(np.array(resized_frame, dtype=np.float64)/255.0)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# getting all the training frames in numpy array
all_frames = np.array(all_frames)
all_frames.shape

# preparing data for training
def get_clips_sequence(stride, frames_list, sequence_size):
    clips = []
    sz = len(frames_list)
    # print(sz)
    clip = np.zeros(shape=(sequence_size, 128, 128, 3))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, :] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips

# preparing data for training
def dataloader(all_frames):
    clips = []
    for stride in range(1, 3):
        clips.extend(get_clips_sequence(stride=stride, frames_list=all_frames, sequence_size=7))
    return clips

training_set = dataloader(all_frames)

training_set = np.asarray(training_set)
training_set = training_set.reshape(-1, 7, 128, 128, 3)

print(training_set.shape)

# Generator model
# Encoder 1
input_layer = Input(shape=(7, 128, 128, 3))
x = TimeDistributed((SeparableConv2D(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2')))(input_layer)
x = BatchNormalization()(x)
sc1 = tanh(x)
x = TimeDistributed(SeparableConv2D(64, (3, 3), strides=2, padding="same", kernel_regularizer='l2'))(sc1)
x = BatchNormalization()(x)
sc2 = tanh(x)
x = ConvLSTM2D(64, (3, 3), padding="same", strides=2, return_sequences=True, kernel_regularizer='l2')(sc2)
x = BatchNormalization()(x)
x = tanh(x)
x = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(x)
x = BatchNormalization()(x)
sc3 = tanh(x)
x = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(sc3)
x = BatchNormalization()(x)
x = tanh(x)
x = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(x)
x = BatchNormalization()(x)
x = tanh(x)
encoder1 = Model(inputs=input_layer, outputs=x)

encoder1.summary()

# Decoder
y = encoder1(input_layer)
y = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
y = keras.layers.Add()([sc2, y])
y = TimeDistributed(Conv2DTranspose(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
y = keras.layers.Add()([sc1, y])
y = TimeDistributed(Conv2DTranspose(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
y = TimeDistributed(SeparableConv2D(3, (5, 5), activation="tanh", padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
g = Model(inputs=input_layer, outputs=y)

g.summary()

# Encoder 2
input_layer = Input(shape=(7, 128, 128, 3))
z = TimeDistributed((SeparableConv2D(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2')))(input_layer)
z = BatchNormalization()(z)
z = tanh(z)
z = TimeDistributed(SeparableConv2D(64, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(64, (3, 3), padding="same", strides=2, return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
encoder2 = Model(inputs=input_layer, outputs=z)



# Discriminator model
input_layer = Input(shape=(7, 128, 128, 3))
f = TimeDistributed((SeparableConv2D(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2')))(input_layer)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = TimeDistributed(SeparableConv2D(64, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(64, (3, 3), padding="same", strides=2, return_sequences=True, kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True,kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
feature_extractor = Model(inputs=input_layer, outputs=f)

feature_extractor.summary()

# Adversial loss
class AdvLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

# Contextual loss
class ContLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ContLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

# Encoder loss
class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(encoder1(ori) - encoder2(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

gan = g(input_layer)
adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cont_loss = ContLoss(name='cont_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cont_loss, enc_loss])

# loss function
def loss(yt, yp):
    return yp

losses = {
    'adv_loss': loss,
    'cont_loss': loss,
    'enc_loss': loss
}

lossWeights = {'cont_loss': 15.0, 'adv_loss': 1.0, 'enc_loss': 1.0}

gan_trainer.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)

# discriminator model
f = feature_extractor(input_layer)
d = TimeDistributed(GlobalAveragePooling2D(name='glb_avg'))(f)
d = TimeDistributed(Dense(1, activation='sigmoid', name='d_out'))(d)
d = Model(input_layer, d)
opt = RMSprop(learning_rate=0.00005)
d.compile(optimizer=opt, loss='binary_crossentropy')

batch = 1

# generate training samples in batch
def get_data_generator(data, batch_size=1):
    
    datalen = len(data)
    cnt = 0
    while True:
        index = np.arange(datalen)
        
        np.random.shuffle(index)
        cnt += 1
        for i in range(int(np.ceil(datalen / batch_size))):
            train_x = np.take(data, index[i * batch_size: (i + 1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y,y]

train_data_generator = get_data_generator(training_set, batch)

sample_display_image = training_set[:2]

# function to display generated images during training
def plot_save(count, g):
    gan_x = g.predict(sample_display_image)
    test_image = gan_x[0, 2, :, :, :]
    test_image = np.reshape(test_image, (128, 128,3))

    minv=np.min(test_image)
    maxv=np.max(test_image)
    new_image=(test_image-minv)/(maxv-minv)
    filename = image_dir + "Reconstructed_image" + str(count) + ".png"
    if new_image.min()>=0.0 and new_image.max() <=1.0:
        plt.imsave(filename, new_image)
    else:
        print(new_image)

d_loss_history = []
generator_loss=[]
print(type(d_loss_history))

# training the GAN model for 5000 epochs
for i in range(5001):

    x, y = train_data_generator.__next__()
    
    d.trainable = True

    generated_x = g.predict(x)

    d_x = np.concatenate([x, generated_x], axis=0)
    a = np.zeros((len(x), 7, 1))
    b = np.ones((len(generated_x), 7, 1))
    d_y = np.concatenate([a, b])
    d_loss = d.train_on_batch(d_x, d_y)


    d.trainable = False
    g_loss = gan_trainer.train_on_batch(x, y)

    if i % 1000 == 0:
        model_name = model_dir + 'g' + str(i) + '.h5'
        # saving the model (encoder-decoder) architecture
        g.save(model_name)

        plot_save(i, g)

        # Storing the generator loss
        generator_loss.append(g_loss[0]+g_loss[1]+g_loss[2]+g_loss[3])

        ## Storing the discriminator loss
        d_loss_history.append(d_loss)



d_loss_history = np.array(d_loss_history)

epoch=[0,1000,2000,3000,4000,5000]

print(d_loss_history)
print(generator_loss)


plt.plot(epoch, generator_loss, label = "generator loss")
plt.plot(epoch, d_loss_history, label = "discriminator loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss for Shanghai Tech Dataset')
# show a legend on the plot
plt.legend()
# Save plot
plt.savefig('training_loss_ShanghaiTech.png', dpi=300, bbox_inches='tight')
# plt.savefig('training_loss_Avenue.png', dpi=300, bbox_inches='tight')
# plt.savefig('training_loss_UCSDped1.png', dpi=300, bbox_inches='tight')
# plt.savefig('training_loss_UCSDped2.png', dpi=300, bbox_inches='tight')