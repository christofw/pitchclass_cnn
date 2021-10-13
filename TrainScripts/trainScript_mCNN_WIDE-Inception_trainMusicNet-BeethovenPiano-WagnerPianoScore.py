import numpy as np
import os

import tensorflow as tf
from random import shuffle

from tensorflow.keras.optimizers import Adam

import datetime
import LibFMP.B
from utils_DL import autoTrain, weightedBinaryCrossentropy
import random
from customModels import musicalCNN_Jo_lastConv_maxPool_LN_inception


########### MODEL ######################################################
model = musicalCNN_Jo_lastConv_maxPool_LN_inception(B=3, 
                                            C=5, 
                                            L=75, 
                                            numOctaves=6, 
                                            configLayer1=[[25,(3,3)], [25,(9,9)], [25,(15,15)], [25,(27,27)]], 
                                            numFilt_filt2=100, 
                                            size2_filt2=3, stride2_filt2=3, 
                                            numFilt_filt3=50, 
                                            numFilt_filt4=5, 
                                            dropout=0.2)

model.summary()

opt = Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=['accuracy','binary_accuracy','binary_crossentropy','cosine_similarity','Precision','Recall'], 
            run_eagerly=False)

modelName = datetime.datetime.now().strftime(format = "%Y-%m-%d_%H-%M-%S") + 'mCNN_WIDE-Inception_trainMusicNet-BeethovenPiano-WagnerPianoScore'
###################################################################################
numValFiles = 100

musicNetFolder = '../../../Datasets/MusicNet_tuning_50_snips'
musicNetFiles =  [f for f in os.listdir(os.path.join(musicNetFolder, 'Chroma')) if '.npy' in f]
shuffle(musicNetFiles)
musicNetTrainFiles = musicNetFiles[numValFiles:]
musicNetValFiles = musicNetFiles[:numValFiles]

beethovenFolder = '../../../Datasets/BeethovenPiano_tuning_50_snips'
beethovenFiles =  [f for f in os.listdir(os.path.join(beethovenFolder, 'Chroma')) if '.npy' in f]
shuffle(beethovenFiles)
beethovenTrainFiles = beethovenFiles[numValFiles:]
beethovenValFiles = beethovenFiles[:numValFiles]

wagnerFolder = '../../../Datasets/WagnerRing_PianoScore_tuning_50_snips'
wagnerFiles =  [f for f in os.listdir(os.path.join(wagnerFolder, 'Chroma')) if '.npy' in f]
shuffle(wagnerFiles)
wagnerTrainFiles = wagnerFiles[numValFiles:]
wagnerValFiles = wagnerFiles[:numValFiles]


trainData = {'files': [musicNetTrainFiles, beethovenTrainFiles, wagnerTrainFiles],
             'pathHCQT': [os.path.join(musicNetFolder, 'HCQT'),
                         os.path.join(beethovenFolder, 'HCQT'),
                         os.path.join(wagnerFolder, 'HCQT')],
             'pathChroma': [os.path.join(musicNetFolder, 'Chroma'),
                         os.path.join(beethovenFolder, 'Chroma'),
                         os.path.join(wagnerFolder, 'Chroma')]}

valData = {'files': [musicNetValFiles, beethovenValFiles, wagnerValFiles],
             'pathHCQT': [os.path.join(musicNetFolder, 'HCQT'),
                         os.path.join(beethovenFolder, 'HCQT'),
                         os.path.join(wagnerFolder, 'HCQT')],
             'pathChroma': [os.path.join(musicNetFolder, 'Chroma'),
                         os.path.join(beethovenFolder, 'Chroma'),
                         os.path.join(wagnerFolder, 'Chroma')]}


##################################################################################

autoTrain(model, modelName, trainData, valData, max_epochs=200, steps_per_epoch=4000, batchSize_train=25, batchSize_val=50, lr_decay=0.2, lr_min=1e-6, lr_patience=3, earlyStopping_patience=8, criterion='val_loss', criterion_mode='min', log=True)
