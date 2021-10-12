######################################################
#             KERAS DATA GENERATORS                  #
#                                                    #
# generate CNN input batches for training,           #
#  validation and testing from precomputed HCQTs     #
#                                                    #
# Johannes Zeitler (johannes.zeitler@fau.de)         #
# Dec. 2020                                          #
######################################################

import numpy as np
import tensorflow as tf
import os
import random

# outputs batches of HCQTs + chroma labels. Supports multiple folders with input data (i.e. multiple datasets)
# Assumption: for each audio sample, HCQTs and chromagrams are stored in 2 separate folders. The filenames are identical.

# input arguments
#  fileList: a list of fileLists for each input folder [fileList_folder1, fileList_folder2...]. Input files are songs (or parts of songs)
#  CQT_directory: a list of the folders containing HCQTs [hcqt_folder1, hcqt_folder2,...]
#  chroma_directory: a list of the folders containing chromagrams [chroma_folder1, chroma_folder2]
#  batch_size: size of output batches
#  numContextFrames: number of context frames
#  compression: factor for log-compression
#  shuffle: whether to shuffle file order at the end of an epoch
#  balanceSets: if true, draw an equivalent number of files from each folder in each epoch (to balance datasets)

# output data:
#  __getitem__ returns a batch of data with dimension (batch_size x nCqtBins x nContextFrames x nChannels)

# procedure:
#  each batch contains samples from only one input file (i.e. one song). The context segments are drawn randomly from the complete input file, i.e. they may be overlapping. 
class TrainGenerator(tf.keras.utils.Sequence):
    def __init__(self, fileList, CQT_directory, chroma_directory, batch_size=25, numContextFrames=75, compression=10, shuffle=True, balanceSets=True): 
        # number of context frames between start/end frame and center frame
        self.oneSideContext = int((numContextFrames-1)/2)
        
        self.fileList = fileList
        self.CQT_directory = CQT_directory
        self.chroma_directory = chroma_directory
        self.balanceSets = balanceSets
        
        if balanceSets:
            cqtList, chromaList = self.__buildLists()
        else:
            cqtList = []
            chromaList = []
            for i in range(len(CQT_directory)):
                cq = CQT_directory[i]
                ch = chroma_directory[i]
                files = fileList[i]
                for f in files:
                    cqtList.append(os.path.join(cq, f))
                    chromaList.append(os.path.join(ch, f))
                
        self.cqtList = cqtList
        self.chromaList = chromaList
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.compression = compression
        self.numContextFrames = numContextFrames
        self.on_epoch_end()        
    
    # draw equally many train files from multiple datasets into a single list
    def __buildLists(self):        
        minLen = 1e20
        for l in self.fileList:
            if len(l) < minLen:
                minLen = len(l)
                
        cqtList = []
        chromaList = []
        for i in range(len(self.CQT_directory)):
            cq = self.CQT_directory[i]
            ch = self.chroma_directory[i]
            files = self.fileList[i]
            for f in random.sample(files, minLen):
                cqtList.append(os.path.join(cq, f))
                chromaList.append(os.path.join(ch, f))
        
        print('\n built lists with length %i'%(len(cqtList)))
        return cqtList, chromaList
    
    # return the total number of batches
    def __len__(self):
        return len(self.cqtList)
    
    # get one batch of data, specified by index
    def __getitem__(self, index):
        return self.__data_generation(index)
                
    # generate a batch of data
    def __data_generation(self, index):
        
        # load cqt and chroma data for one file
        cqt = np.load(self.cqtList[index])
        chroma = np.load(self.chromaList[index])
        
        # apply log. compression
        if self.compression is not None:
            cqt = np.log(1+self.compression*cqt)
        
        # randomly choose center indices of the context frames
        centerInds= np.random.randint(low=self.oneSideContext+1,
                                      high=chroma.shape[1]-self.oneSideContext-1,
                                      size=self.batch_size)  
        
        # the lists that will contain all elements of a batch
        data = []
        labels = []
        
        # generate the elements of a batch
        for ind in centerInds:
            # 'cut out' a context window surrounding the target frame
            cqtFrame = cqt[:,ind-self.oneSideContext:ind+self.oneSideContext+1,:]
            # pick the target frame
            chromaFrame = chroma[:,ind]

            data.append(cqtFrame)
            labels.append(chromaFrame)
        
        # reshape data_ and labels_ to match the dimension requirements of a batch (batchSize x numCQTbins x numContext x numChannels)
        data_ = np.asarray(data).reshape(-1, 
                                         cqtFrame.shape[0], 
                                         self.numContextFrames, 
                                         cqtFrame.shape[-1])
        
        labels_ = np.asarray(labels).reshape(-1, 
                                             chroma.shape[0])
        
        # output a batch of data for model training
        return data_, labels_

    # on epoch end, draw new files from the datasets and shuffle the lists
    def on_epoch_end(self):
        if self.balanceSets:
            self.cqtList, self.chromaList = self.__buildLists()
            
        if self.shuffle:
            print('\n shuffle train set...\n')
            temp = list(zip(self.cqtList, self.chromaList)) 
            random.shuffle(temp) 
            res1, res2 = zip(*temp)
            self.cqtList = list(res1)
            self.chromaList = list(res2)


# same as TrainGenerator, only for validation purpose (no shuffling, context segments are always picked from the same equally spaced locations)
class ValidationGenerator(tf.keras.utils.Sequence):
    def __init__(self, fileList, CQT_directory, chroma_directory, batch_size=50, numContextFrames=75, compression=10, shuffle=False):
        
        self.oneSideContext = int((numContextFrames-1)/2)
        
        cqtList = []
        chromaList = []
        for i in range(len(CQT_directory)):
            cq = CQT_directory[i]
            ch = chroma_directory[i]
            files = fileList[i]
            for f in files:
                cqtList.append(os.path.join(cq, f))
                chromaList.append(os.path.join(ch, f))
                
        self.cqtList = cqtList
        self.chromaList = chromaList
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.compression = compression
        self.numContextFrames = numContextFrames
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.cqtList)

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(index)
        
    
    def __data_generation(self, index):
        
        cqt = np.load(self.cqtList[index])
        chroma = np.load(self.chromaList[index])
        
        if self.compression is not None:
                cqt = np.log(1+self.compression*cqt)
        
        centerInds = np.linspace(self.oneSideContext+1,chroma.shape[1]-self.oneSideContext-1,self.batch_size).astype(int)
        
        data = []
        labels = []
        
        for ind in centerInds:
            cqtFrame = cqt[:,ind-self.oneSideContext:ind+self.oneSideContext+1,:]
            chromaFrame = chroma[:,ind]

            data.append(cqtFrame)
            labels.append(chromaFrame)
            
        data_ = np.asarray(data).reshape(-1, 
                                         cqtFrame.shape[0], 
                                         self.numContextFrames, 
                                         cqtFrame.shape[-1])
        
        labels_ = np.asarray(labels).reshape(-1, 
                                             chroma.shape[0])
        
        return data_, labels_

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.cqtList, self.chromaList)) 
            random.shuffle(temp) 
            res1, res2 = zip(*temp)
            self.cqtList = list(res1)
            self.chromaList = list(res2)

            
# Generator for predicting chromas for a single song. Stride is one, i.e. a chroma vector is predicted for each input frame.

# input arguments:
#  cqt: precomputed hcqt for a certain song
#  batch_size: size of the output batches. A larger batch size usually leads to faster processing. However, the choice of batch size doesn't influence the network predictions.
class PredictGenerator(tf.keras.utils.Sequence):
    def __init__(self, cqt, batch_size=100, numContextFrames=75, compression=10):
        
        self.oneSideContext = int((numContextFrames-1)/2)
        self.batch_size = batch_size
        self.numContextFrames = numContextFrames
        
        #zero-padding
        cqt = np.pad(cqt, ((0,0), (self.oneSideContext, self.oneSideContext), (0,0)), mode='constant')
        
        #log-compression
        if compression is not None:
            cqt = np.log(1+compression*cqt)
        
        self.cqt = cqt        
        self.cqtLen = cqt.shape[1]

    def __len__(self):
        return np.floor(self.cqtLen/self.batch_size).astype(int)

    def __getitem__(self, index):
        return self.__data_generation(index)
    
    def __data_generation(self, index):
        
        startInd = np.max( (index*self.batch_size, self.oneSideContext))
        stopInd = np.min(( (index+1)*self.batch_size, self.cqtLen - self.oneSideContext))
        
        centerInds = np.arange(startInd, stopInd)
        
        data = []
        
        for ind in centerInds:
            cqtFrame = self.cqt[:,ind-self.oneSideContext:ind+self.oneSideContext+1,:]            
            data.append(cqtFrame)
            
        data_ = np.asarray(data).reshape(-1, 
                                         cqtFrame.shape[0], 
                                         self.numContextFrames, 
                                         cqtFrame.shape[-1])        
        return data_

    def on_epoch_end(self):
        return 0