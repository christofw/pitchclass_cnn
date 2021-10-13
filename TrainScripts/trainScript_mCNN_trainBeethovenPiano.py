import numpy as np
import os

import tensorflow as tf
from random import shuffle

from tensorflow.keras.optimizers import Adam

import datetime
import LibFMP.B
from utils_DL import autoTrain, weightedBinaryCrossentropy
import random
from customModels import musicalCNN_Jo_lastConv_maxPool_LN


########### MODEL ######################################################
model = musicalCNN_Jo_lastConv_maxPool_LN(B=3, 
                       C=5, 
                       L=75, 
                       numOctaves=6, 
                       numFilters=[20,20,10,1], 
                       size_filt1=(15,15), 
                       size2_filt2=3, stride2_filt2=3, 
                       dropout=0.2,
                       alpha=0.3)
model.summary()

opt = Adam(learning_rate=0.005)

model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=['accuracy','binary_accuracy','binary_crossentropy','cosine_similarity','Precision','Recall'], 
            run_eagerly=False)

modelName = datetime.datetime.now().strftime(format = "%Y-%m-%d_%H-%M-%S") + 'mCNN_trainBeethovenPiano'
###################################################################################
numValFiles = 200

trainFolder = '../../../Datasets/BeethovenPiano_tuning_50_snips'
allFiles =  [f for f in os.listdir(os.path.join(trainFolder, 'Chroma')) if '.npy' in f]
shuffle(allFiles)

trainFiles = allFiles[numValFiles:]
valFiles = allFiles[:numValFiles]


trainData = {'files': [trainFiles],
             'pathHCQT': [os.path.join(trainFolder, 'HCQT')],
             'pathChroma': [os.path.join(trainFolder, 'Chroma')]}

valData = {'files': [valFiles],
             'pathHCQT': [os.path.join(trainFolder, 'HCQT')],
             'pathChroma': [os.path.join(trainFolder, 'Chroma')]}


##################################################################################

autoTrain(model, modelName, trainData, valData, max_epochs=200, steps_per_epoch=4000, batchSize_train=25, batchSize_val=50, lr_decay=0.2, lr_min=1e-6, lr_patience=3, earlyStopping_patience=8, criterion='val_loss', criterion_mode='min', log=True)
