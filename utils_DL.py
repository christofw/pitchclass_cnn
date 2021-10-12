######################################################
#             DL UTILITY FUNCTIONS                   #
#                                                    #
# Some Deep-Learning related utility functions for   #
#  deep chroma estimation                            #
#                                                    #
# Johannes Zeitler (johannes.zeitler@fau.de)         #
# Dec. 2020                                          #
######################################################

import numpy as np
import random
import tensorflow as tf
import LibFMP.B
import LibFMP.C3
import matplotlib.pyplot as plt
import os
import tensorflow.keras.backend as KB
from tensorflow.python.framework import ops
from scipy import ndimage as scImage
import csv
import copy
import FrameGenerators
from tqdm import tqdm


# automatically train and save a Neural Network. Learning rate is automatically reduced on plateaus, early stopping if no further improvements are monitored.
#
# Input arguments:
#  model: a compiled neural network
#  modelName: model save name
#  trainData: a dictionary containing fields
#             'files': a list of fileLists for each input folder [fileList_folder1, fileList_folder2...]. Input files are songs (or parts of songs)
#             'pathHCQT': a list of the folders containing HCQTs [hcqt_folder1, hcqt_folder2,...]
#             'pathChroma': a list of the folders containing chromagrams [chroma_folder1, chroma_folder2]
#  valData: similar dictionary as trainData, only for validation purposes (lr-decay, early stopping)
#  
def autoTrain(model, modelName, trainData, valData, max_epochs=200, steps_per_epoch=None, batchSize_train=25, batchSize_val=25, lr_initial=0.01, lr_decay=0.2, lr_min=1e-6, lr_patience=3, earlyStopping_patience=8, criterion='val_loss', criterion_mode='max', log=True):
    
    # build training and validation data generators
    trainGenerator = FrameGenerators.TrainGenerator(fileList=trainData['files'], 
                                                       CQT_directory=trainData['pathHCQT'],
                                                       chroma_directory=trainData['pathChroma'], 
                                                       batch_size=batchSize_train, 
                                                       numContextFrames=model.input_shape[2],
                                                       shuffle=True)

    
    if valData is None:
        valGenerator = None
    else:
        valGenerator = FrameGenerators.ValidationGenerator(fileList=valData['files'], 
                                                       CQT_directory=valData['pathHCQT'],
                                                       chroma_directory=valData['pathChroma'], 
                                                       batch_size=batchSize_val, 
                                                       numContextFrames=model.input_shape[2],
                                                       shuffle=False)
    
    # specify lr-decay and early stopping
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=criterion,
                    factor=lr_decay,
                    patience=lr_patience,
                    verbose=1,
                    mode=criterion_mode,
                    min_delta=1e-4,
                    cooldown=0,
                    min_lr=lr_min),
                 
                tf.keras.callbacks.EarlyStopping(
                    monitor=criterion,
                    min_delta=1e-4,
                    patience=earlyStopping_patience,
                    verbose=1,
                    mode=criterion_mode,
                    baseline=None,
                    restore_best_weights=True),
                 
                tf.keras.callbacks.TerminateOnNaN()]
    
    # print training and validation statistics for each epoch
    if log:
        callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join('./logs',modelName+'.txt'), separator=",", append=True))
    
    # train and save the model
    model.fit_generator(generator=trainGenerator, 
                      epochs=max_epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=valGenerator,
                      #validation_steps=50,
                      callbacks=callbacks, 
                      use_multiprocessing=True,
                      workers=4)
    
    model.save('./Models/'+modelName)

# train a model and predict chromagrams afterwards. Used in cross-validation scenarios. Uses legacy frameGenerator() function from utils_DL
def trainAndPredict(model, trainFiles, valFiles, testFiles, CQT_directory, chroma_directory, save_directory, csvFilename, hopSizeCQT, frameRate_out, maxEpochs=200, stepsPerEpoch=2000, trainBatchSize=25, fs=22050, transFrame=[]):
    
    
    trainGenerator = frameGenerator(fileList=trainFiles,
                                    CQT_directory=CQT_directory, 
                                    chroma_directory=chroma_directory, 
                                    batchSize=trainBatchSize,
                                    numContextFrames=model.input_shape[2], 
                                    stride=10, 
                                    transFrame=transFrame,
                                    randomStartIndex=True,
                                    pad=False)
    trainData = tf.data.Dataset.from_generator(lambda:trainGenerator,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(model.input_shape,(None,12)))
    
    #trainData = trainData.shuffle(buffer_size=1000)
    
    if len(valFiles) > 0:
        valGenerator = frameGenerator(fileList=valFiles,
                                        CQT_directory=CQT_directory, 
                                        chroma_directory=chroma_directory, 
                                        batchSize=50,
                                        numContextFrames=model.input_shape[2], 
                                        stride=25, 
                                        transFrame=transFrame,
                                        randomStartIndex=True,
                                        pad=False)
        valData = tf.data.Dataset.from_generator(lambda:valGenerator,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(model.input_shape,(None,12)))
    else:
        valData = None
    
    callbacks = []
    '''
        #tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_cosine_similarity",
            factor=0.2,
            patience=3,
            verbose=1,
            mode="max",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_cosine_similarity",
            min_delta=0,
            patience=5,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=False),
        #tf.keras.callbacks.CSVLogger('trainLogs.txt', separator=",", append=True)]'''
    
    model.fit(trainData.cache(), epochs=maxEpochs, validation_data=valData, callbacks=callbacks)
    
    print('Start testing')
    for testFile in testFiles:
        testGenerator = frameGenerator(fileList=[testFile],
                                    CQT_directory=CQT_directory, 
                                    chroma_directory=chroma_directory, 
                                    batchSize=500,
                                    numContextFrames=model.input_shape[2], 
                                    stride=1, 
                                    transFrame=transFrame,
                                    randomStartIndex=False,
                                    pad=False)
        testData = tf.data.Dataset.from_generator(lambda:testGenerator,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(model.input_shape,(None,12)))
        
        predictGenerator = frameGenerator(fileList=[testFile],
                                    CQT_directory=CQT_directory, 
                                    chroma_directory=chroma_directory, 
                                    batchSize=500,
                                    numContextFrames=model.input_shape[2], 
                                    stride=1, 
                                    transFrame=transFrame,
                                    randomStartIndex=False,
                                    pad=True)
        predictData = tf.data.Dataset.from_generator(lambda:predictGenerator,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(model.input_shape,(None,12)))
        
        # evaluate
        evalMetrics = model.evaluate(testData)
        fields = [testFile.split('.')[0]]
        for m in evalMetrics:
            fields.append(m)
        with open(os.path.join(save_directory, csvFilename), 'a+', newline='') as csvfile:  
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)  
            # writing the fields  
            csvwriter.writerow(fields)
        
        #predict
        output = model.predict(predictData).transpose()         
        cqtTimes = np.arange(output.shape[1]) / (fs/hopSizeCQT)
        targetTimes = np.arange(0, np.max(cqtTimes), 1/frameRate_out)        
        outputInterp = np.zeros((12, len(targetTimes)))
        for i in range(12):
            outputInterp[i,:] = np.interp(targetTimes, cqtTimes, output[i,:])        
        np.save(os.path.join(save_directory, testFile), outputInterp)

# extract the first C channels from harmonic CQT        
def extractHCQTchannels(cq, ch, C):
    cq_ = np.zeros((cq.shape[0], cq.shape[1], C))
    cq_[:,:,:] = cq[:,:,:C]
    return cq_, ch 
    

# weighted Binary Cross-Entropy loss function as described in report 
def weightedBinaryCrossentropy(zerosWeight=.5):
    def loss(y_true, y_pred):
        eps = KB.epsilon()
        onesWeight = 1 - zerosWeight
        corrFac = 2
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = ops.convert_to_tensor_v2(y_true)
        
        y_pred = KB.clip(y_pred, eps, 1-eps)
        bce = y_true*KB.log(y_pred)*onesWeight*corrFac
        bce += (1 - y_true) * KB.log(1 - y_pred)*zerosWeight*corrFac
        return KB.mean(-bce, axis=-1)    
    return loss

# pitch-shift to avoid overfitting
def pitchShift(cqt, chroma, rng=[-1, 1], binsPerKey = 3):
    shift = random.randint(rng[0], rng[1])
    if shift == 0:
        return cqt, chroma
    
    cqt_ = np.zeros_like(cqt)
    if shift > 0:        
        cqt_[shift*binsPerKey:,:,:] = cqt[:-shift*binsPerKey,:,:]    
    else:
        cqt_[:-shift*binsPerKey] = cqt[shift*binsPerKey:]
    
    return cqt_, np.roll(chroma,shift)


# predict chromagrams and downsample (default: 10Hz output rate)
def predictAndSafe_downsample(model, testFiles, CQT_directory, save_directory,  fs=22050, hopSize=441, intermediateRate=50, filt_len=5, down_sampling=5, normalize=True, norm='2', overrideInputShape=None):     
    if not os.path.isdir(save_directory):
        print('create directory...')
        os.mkdir(save_directory)
    print('writing to '+save_directory)
    
    if overrideInputShape is None:
        inputShape = model.input_shape
    else:
        inputShape = overrideInputShape
    
    for testFile in tqdm(testFiles):
        testGen = testGenerator(fileList=[testFile], 
                                       CQT_directory=CQT_directory, 
                                       batchSize=None,
                                       numContextFrames=inputShape[2], 
                                       stride=1,
                                       pad=True,
                                       padMode='reflect')
        testDataset = tf.data.Dataset.from_generator(lambda:testGen,
                                          output_types=tf.float32,
                                          output_shapes=inputShape)

        

        output = model.predict(testDataset).transpose() 
        
        cqtTimes = np.arange(output.shape[1]) / (fs/hopSize)
        targetTimes = np.arange(0, np.max(cqtTimes), 1/intermediateRate)
        
        outputInterp = np.zeros((12, len(targetTimes)))
        for i in range(12):
            outputInterp[i,:] = np.interp(targetTimes, cqtTimes, output[i,:])
            
        outputDownsample,_ = LibFMP.C3.smooth_downsample_feature_sequence(outputInterp, fs, filt_len=filt_len, down_sampling=down_sampling, w_type='boxcar')
        
        if normalize:
            outputDownsample = LibFMP.C3.normalize_feature_sequence(outputDownsample, norm=norm, threshold=0.0001, v=None)
        
        np.save(os.path.join(save_directory, testFile), outputDownsample)

# predict chromagrams and store them
def predictAndSafe(model, testFiles, CQT_directory, save_directory,  fs=22050, hopSize=441, desiredRate=50):    
    for testFile in testFiles:
        print('processing '+testFile)
        testGen = testGenerator(fileList=[testFile], 
                                       CQT_directory=CQT_directory, 
                                       batchSize=None,
                                       numContextFrames=model.input_shape[2], 
                                       stride=1,
                                       pad=True,
                                       padMode='reflect')
        testDataset = tf.data.Dataset.from_generator(lambda:testGen,
                                          output_types=tf.float32,
                                          output_shapes=model.input_shape)
        

        output = model.predict(testDataset).transpose() 
        print(output.shape)
        
        cqtTimes = np.arange(output.shape[1]) / (fs/hopSize)
        targetTimes = np.arange(0, np.max(cqtTimes), 1/desiredRate)
        
        outputInterp = np.zeros((12, len(targetTimes)))
        for i in range(12):
            outputInterp[i,:] = np.interp(targetTimes, cqtTimes, output[i,:])        
        
        np.save(os.path.join(save_directory, testFile), outputInterp)

# predict a bunch of chromagrams and sonify them
def predictAndSonify(model, testFile, CQT_directory, chroma_directory, limTime=None, fs=1, hopSize=1, numBins=216, L=15, C=5):

    testDataset=tf.data.Dataset.from_generator(frameGenerator, args=[[testFile], CQT_directory, chroma_directory, 1, L, 1], 
                                output_types=(tf.float32, tf.float32),
                               output_shapes=((None,numBins,L,C),(None,12)))

    outputRaw = model.predict(testDataset) #> 0.5
    output = outputRaw > 0.5
    label = []
    for _, chroma in testDataset.take(output.shape[0]):
        label.append(chroma.numpy())

    label = np.asarray(label).reshape(output.shape)
    
    if limTime is not None:
        label = label[np.max((0, int(limTime[0]*fs/hopSize))):np.min((label.shape[0], int(limTime[1]*fs/hopSize))),:]
        output = output[np.max((0, int(limTime[0]*fs/hopSize))):np.min((output.shape[0], int(limTime[1]*fs/hopSize))),:]
        outputRaw = outputRaw[np.max((0, int(limTime[0]*fs/hopSize))):np.min((outputRaw.shape[0], int(limTime[1]*fs/hopSize))),:]

    numSamples = output.shape[0]*hopSize
    output_son = LibFMP.B.sonify_chromagram(output.transpose(), N=numSamples, frame_rate=fs/hopSize, Fs=fs, fading_msec=5)
    outputRaw_son = LibFMP.B.sonify_chromagram(outputRaw.transpose(), N=numSamples, frame_rate=fs/hopSize, Fs=fs, fading_msec=5)
    label_son = LibFMP.B.sonify_chromagram(label.transpose(), N=numSamples, frame_rate=fs/hopSize, Fs=fs, fading_msec=5)
    return output_son, outputRaw_son, label_son

# visualize predicted chromagrams and illustrate color-coded TPs, TNs, FPs, FNs
def visualizeTestSet(model, testFiles, CQT_directory, chroma_directory, fs=1, hopSize=1, figsize=(8,4), sonify=False,
                    numBins=216, L=15, C=5):
    sonifications=[]
    for i in range(len(testFiles)):
        f = testFiles[i]
        print('-----------------------------------------------')
        print('Inference on ',f[:-4])
        testDataset=tf.data.Dataset.from_generator(frameGenerator, args=[[f], CQT_directory, chroma_directory, 1, L, 1], 
                                    output_types=(tf.float32, tf.float32),
                                   output_shapes=((None,numBins,L,C),(None,12)))

        outputRaw = model.predict(testDataset) #> 0.5
        output = outputRaw > 0.5
        label = []
        for _, chroma in testDataset.take(output.shape[0]):
            label.append(chroma.numpy())

        label = np.asarray(label).reshape(output.shape)
        print('%.2f < label %.2f, %.2f < outputRaw < %.2f'%(np.min(label), np.max(label), np.min(outputRaw), np.max(outputRaw)))

    
        col_truePos=[0,0,0]
        col_trueNeg=[1,1,1]
        col_falsePos=[1,0,0]
        col_falseNeg=[.5,0,0]

        error_RGB = np.zeros((output.shape[0], output.shape[1], 3))
        #true pos -> black
        mask_tp = output*label
        #true neg -> white
        mask_tn = np.abs(output-1)*np.abs(label-1)
        #false pos -> red
        mask_fp = output*np.abs(label-1)
        #false neg -> blue
        mask_fn = np.abs(output-1)*label

        error_RGB[:,:,0] = mask_fp + mask_tn
        error_RGB[:,:,1] = mask_tn
        error_RGB[:,:,2] = mask_tn + mask_fn

        T = np.arange(output.shape[0])*hopSize/fs

        LibFMP.B.plot_chromagram(np.swapaxes(error_RGB, 0,1), interpolation='none', 
                                 title='CNN errors: TP (black), FP (red), FN (blue)', colorbar=None,
                                figsize=figsize, T_coef=T)
        plt.show()
        precision = np.sum(mask_tp)/np.sum(mask_tp + mask_fp)
        recall = np.sum(mask_tp)/np.sum(mask_tp + mask_fn)
        fMeasure = 2*precision*recall/(precision+recall)
        print('F1 score: %.3f, precision: %.3f, recall: %.3f'%(fMeasure,precision,recall))        
        
        if sonify:
            numSamples = output.shape[0]*hopSize
            output_son = LibFMP.B.sonify_chromagram(output, N=numSamples, frame_rate=hopSize, Fs=fs, fading_msec=5)
            outputRaw_son = LibFMP.B.sonify_chromagram(outputRaw, N=numSamples, frame_rate=hopSize, Fs=fs, fading_msec=5)
            label_son = LibFMP.B.sonify_chromagram(label, N=numSamples, frame_rate=hopSize, Fs=fs, fading_msec=5)
            sonifications.append({'filename': f,
                                 'outputRaw_son': outputRaw_son,
                                 'outputDiscr_son':output_son,
                                 'label_son':label_son})
        
        print('-----------------------------------------------')
    return sonifications

# remove all elements from a list of files, that contain at least one entry in 'strings'. Used for e.g. doing an artist split.
def removeStrFromList(files, strings):
    listOut = []
    listIn = []
    for f in files:
        match = False
        for s in strings:
            if s in f:
                match=True
                listIn.append(f)
                break
        if not match:
            listOut.append(f)
    return listOut, listIn


##################################################################################
########################## LEGACY FRAME GENERATORS ###############################
# Frame generators used in early development and for some prediction methods.
# Work only if the complete dataset fits into memory
#
# Common input arguments
#  fileList: a list of files to load in the generator. Input files are songs (or parts of songs)
#  CQT_directory: folder containing HCQTs 
#  chroma_directory: folder containing ground-truth chromagrams 
#  batch_size: size of output batches
#  numContextFrames: number of context frames
#  compression: factor for log-compression
#  stride: spacing of the central indices of context frames.
#  transFile: a list of functions that are executed on a file-wise basis
#  transFrame: a list of functions that are executed on a frame-wise basis (i.e. a segment with context)
#  randomStartIndex: if true, pick the context frames at random locations

# Frame generator for multiple datasets
def frameGenerator_multiSets(fileList, CQT_directory, chroma_directory, batchSize=25, numContextFrames=15, stride=5, compression=10, transFile=[], transFrame=[], randomStartIndex=False, shuffleFiles=True, pad=False, padMode='reflect'):
    
    if np.mod(numContextFrames, 2) < 1e-15:
        raise ValueError('Only odd numbers of context frames are allowed')
    oneSideContext = int((numContextFrames-1)/2)
    data = []
    labels = []
    
    cqtList = []
    chromaList = []
    
    for i in range(len(CQT_directory)):
        cq = CQT_directory[i]
        ch = chroma_directory[i]
        files = fileList[i]
        
        for f in files:
            cqtList.append(os.path.join(cq, f))
            chromaList.append(os.path.join(ch, f))
            
    while True:
        if shuffleFiles:
            temp = list(zip(cqtList, chromaList)) 
            random.shuffle(temp) 
            res1, res2 = zip(*temp)
            cqtList = list(res1)
            chromaList = list(res2)
            
        for f_cq, f_ch in zip(cqtList, chromaList):
            cqt = np.load(f_cq)
            chroma = np.load(f_ch)

            if pad:
                cqt = np.pad(cqt, ((0,0), (oneSideContext, oneSideContext), (0,0)), mode=padMode)
                chroma = np.pad(chroma, ((0,0), (oneSideContext, oneSideContext)), mode=padMode)

            for trans in transFile:
                cqt, chroma = trans(cqt, chroma)

            if compression is not None:
                cqt = np.log(1+compression*cqt)

            startInd=0
            if randomStartIndex:
                startInd = random.randint(0, stride)

            centerInds = np.arange(oneSideContext + startInd, cqt.shape[1]-oneSideContext, stride)

            for ind in centerInds:

                cqtFrame = cqt[:,ind-oneSideContext:ind+oneSideContext+1,:]
                chromaFrame = chroma[:,ind]

                for trans in transFrame:
                    cqtFrame, chromaFrame = trans(cqtFrame, chromaFrame)

                data.append(cqtFrame)
                labels.append(chromaFrame)

                if np.abs(len(labels)-batchSize) < 1e-15:
                    data_ = np.asarray(data).reshape(-1, cqtFrame.shape[0], numContextFrames, cqtFrame.shape[-1])
                    labels_ = np.asarray(labels).reshape(-1, chroma.shape[0])
                    data = []
                    labels = []
                    yield data_, labels_

# Frame generator for a single dataset
def frameGenerator(fileList, CQT_directory, chroma_directory, batchSize=25, numContextFrames=15, stride=5, compression=10, transFile=[], transFrame=[], randomStartIndex=False, pad=False, padMode='reflect'):
    
    if np.mod(numContextFrames, 2) < 1e-15:
        raise ValueError('Only odd numbers of context frames are allowed')
    oneSideContext = int((numContextFrames-1)/2)
    data = []
    labels = []
    for f in fileList:
        cqt = np.load(os.path.join(CQT_directory, f))
        chroma = np.load(os.path.join(chroma_directory, f))
        
        if pad:
            cqt = np.pad(cqt, ((0,0), (oneSideContext, oneSideContext), (0,0)), mode=padMode)
            chroma = np.pad(chroma, ((0,0), (oneSideContext, oneSideContext)), mode=padMode)
        
        for trans in transFile:
            cqt, chroma = trans(cqt, chroma)
        
        if compression is not None:
            cqt = np.log(1+compression*cqt)
            
        startInd=0
        if randomStartIndex:
            startInd = random.randint(0, stride)
            
        centerInds = np.arange(oneSideContext + startInd, cqt.shape[1]-oneSideContext, stride)
        
        for ind in centerInds:            
            cqtFrame = cqt[:,ind-oneSideContext:ind+oneSideContext+1,:]
            chromaFrame = chroma[:,ind]
            
            for trans in transFrame:
                cqtFrame, chromaFrame = trans(cqtFrame, chromaFrame)
            
            data.append(cqtFrame)
            labels.append(chromaFrame)
            
            if np.abs(len(labels)-batchSize) < 1e-15:
                data_ = np.asarray(data).reshape(-1, cqtFrame.shape[0], numContextFrames, cqtFrame.shape[-1])
                labels_ = np.asarray(labels).reshape(-1, chroma.shape[0])
                data = []
                labels = []
                yield data_, labels_

# Frame generator for predictions only, no label output
def testGenerator(fileList, CQT_directory, batchSize=1, numContextFrames=15, stride=1, compression=10, transFile=[], transFrame=[], pad=True, padMode='reflect'):
    if np.mod(numContextFrames, 2) < 1e-15:
        raise ValueError('Only odd numbers of context frames are allowed')
    oneSideContext = int((numContextFrames-1)/2)
    data = []    
    if batchSize is not None:
        bs = batchSize
    
    for f in fileList:
        cqt = np.load(os.path.join(CQT_directory, f))
        
        if pad:
            cqt = np.pad(cqt, ((0,0), (oneSideContext, oneSideContext), (0,0)), mode=padMode)
        
        for trans in transFile:
            cqt, cqt = trans(cqt, cqt)
        
        if compression is not None:
            cqt = np.log(1+compression*cqt)
            
            
        centerInds = np.arange(oneSideContext, cqt.shape[1]-oneSideContext, stride)
        
        if batchSize is None:
            bs = np.min([500, len(centerInds)])
        
        for ind in centerInds:
            
            cqtFrame = cqt[:,ind-oneSideContext:ind+oneSideContext+1,:]
            
            for trans in transFrame:
                cqtFrame, cqtFrame = trans(cqtFrame, cqtFrame)
            
            data.append(cqtFrame)
            
            if np.abs(len(data)-bs) < 1e-15:
                data_ = np.asarray(data).reshape(-1, cqtFrame.shape[0], numContextFrames, cqtFrame.shape[-1])
                data = []
                yield data_



##############################################################
##################### DEPRECATED #############################

# get test measures from predicted chromagrams. DEPRECATED, use model.evaluate instead
def getTestMeasures(model, testFiles, CQT_directory, chroma_directory, plot=False, sortPlot=True, figSize=(15,5), numContextFrames=15, fs=1, hopSize=1):
    prec = np.zeros(len(testFiles))
    reca = np.zeros(len(testFiles))
    #fmea = np.zeros(len(testFiles))
    
    for i in range(len(testFiles)):
        f = testFiles[i]
        testDataset=tf.data.Dataset.from_generator(frameGenerator, args=[[f], CQT_directory, chroma_directory, 1], 
                                    output_types=(tf.float32, tf.float32),
                                   output_shapes=((None,216,15,5),(None,12)))

        output = model.predict(testDataset) > 0.5
        label = []
        for _, chroma in testDataset.take(output.shape[0]):
            label.append(chroma.numpy())

        label = np.asarray(label).reshape(output.shape)

        mask_tp = output*label
        #true neg -> white
        #mask_tn = np.abs(output-1)*np.abs(label-1)
        #false pos -> red
        mask_fp = output*np.abs(label-1)
        #false neg -> blue
        mask_fn = np.abs(output-1)*label

        prec[i] = np.sum(mask_tp)/np.sum(mask_tp + mask_fp)
        reca[i] = np.sum(mask_tp)/np.sum(mask_tp + mask_fn)
        #fMeasure = 2*precision*recall/(precision+recall)
    fMeasure = 2*prec*reca/(prec+reca)
    print('F-Measure: mean %.3f, std. %.3f'%(np.mean(fMeasure), np.std(fMeasure)))
    print('Precision: mean %.3f, std. %.3f'%(np.mean(prec), np.std(prec)))
    print('Recall:    mean %.3f, std. %.3f'%(np.mean(reca), np.std(reca)))
    
    if plot:
        if sortPlot:
            sort_inds = np.argsort(fMeasure)
        else:
            sort_inds = np.arange(len(fMeasure))
        
        plt.figure(figsize=figSize)
        plt.plot(fMeasure[sort_inds], linewidth=4, label='F-Measure')
        plt.plot(prec[sort_inds], label='Precision')
        plt.plot(reca[sort_inds], label='Recall')
        plt.xticks(np.arange(len(fMeasure)), [testFiles[a][0:-4] for a in sort_inds], rotation=90)
        plt.legend()
        plt.grid()
        plt.show()
    
    return fMeasure, prec, reca

def contextDataset(fileNames,CQT_directory, chroma_directory, numContextFrames=15, stride=5):    
    f = fileNames[0]
    cqt = np.load(os.path.join(CQT_directory, f))
    chroma = np.load(os.path.join(chroma_directory, f))

    oneSideContext = (int)((numContextFrames-1)/2)
    centerInds = np.arange(oneSideContext, cqt.shape[1]-oneSideContext, stride)
    #print(centerInds)
    #print(cqt.shape)

    audioContext = np.zeros((len(centerInds), cqt.shape[0], numContextFrames, cqt.shape[2]))
    for i in range(len(centerInds)):
        audioContext[i,:,:,:] = cqt[:,centerInds[i]-oneSideContext:centerInds[i]+oneSideContext+1,:]

    chromaContext = chroma[:,centerInds].transpose() > 0
    ds = tf.data.Dataset.from_tensors((audioContext, chromaContext))
    
    if len(fileNames) > 1:
        for f in fileNames[1:-1]:
            cqt = np.load(os.path.join(CQT_directory, f))
            chroma = np.load(os.path.join(chroma_directory, f)) 

            oneSideContext = (int)((numContextFrames-1)/2)
            centerInds = np.arange(oneSideContext, cqt.shape[1]-oneSideContext, stride)
            #print(centerInds)
            #print(cqt.shape)

            audioContext = np.zeros((len(centerInds), cqt.shape[0], numContextFrames, cqt.shape[2]))
            for i in range(len(centerInds)):
                audioContext[i,:,:,:] = cqt[:,centerInds[i]-oneSideContext:centerInds[i]+oneSideContext+1,:]

            chromaContext = chroma[:,centerInds].transpose() > 0
            ds = ds.concatenate(tf.data.Dataset.from_tensors((audioContext, chromaContext)))
            
    return ds