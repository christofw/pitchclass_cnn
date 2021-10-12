######################################################
#            KERAS MODEL DEFINITIONS                 #
#                                                    #
# Johannes Zeitler (johannes.zeitler@fau.de)         #
# Dec. 2020                                          #
######################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU, Flatten, Dense, MaxPooling2D, Dropout, Input, BatchNormalization, LayerNormalization
from tensorflow.keras.models import Sequential

# common input arguments:
#  B: bins per MIDI pitch
#  C: number of HCQT channels
#  L: number of context bins

# 'standard' CNN (referred to as 'default', 'simple', 'wide')
def musicalCNN_Jo_lastConv_maxPool_LN(B, C, L, numOctaves, numFilters=[20,20,10,1], size_filt1=(15,15), size2_filt2=3, stride2_filt2=3, dropout=0.2, alpha=0.3, LNaxis=[1,2,3]):
    cnn = Sequential([
        # Prefiltering
        LayerNormalization(axis=LNaxis, input_shape=(numOctaves*12*B, L, C)),
        Conv2D(numFilters[0], size_filt1, padding='same', name='conv1'),
        LeakyReLU(name='leakyReLU1', alpha=alpha),
        MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid"),
        Dropout(dropout, name='dropout1'),
        
        # Binning to MIDI pitches
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        LeakyReLU(name='leakyReLU2', alpha=alpha),
        MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid"),
        Dropout(dropout, name='dropout2'),
        
        # Time reduction
        Conv2D(numFilters[2], (1,int(L/size2_filt2/4)), name='conv3'),
        LeakyReLU(name='leakyReLU3', alpha=alpha),
        Dropout(dropout, name='dropout3'),
        
        # Chroma reduction
        Conv2D(numFilters[3], (1,1), name='conv4'),
        LeakyReLU(name='leakyReLU4', alpha=alpha),
        Dropout(dropout, name='dropout4'),        
        Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid'),
        Flatten(name='flatten')
    ])    
    return cnn

# 'deep' CNN (prefiltering layer is replicated)
def musicalCNN_Jo_lastConv_maxPool_LN_deep(B, C, L, numOctaves, numFilters=[20,20,10,1], size_filt1=(15,15), size2_filt2=3, stride2_filt2=3, dropout=0.2, alpha=0.3):
    cnn = Sequential([
        LayerNormalization(axis=[1,2,3], input_shape=(numOctaves*12*B, L, C)),
        
        Conv2D(numFilters[0], size_filt1, padding='same', name='conv1_0'),
        LeakyReLU(name='leakyReLU1_0', alpha=alpha),
        Dropout(dropout, name='dropout1_0'),
        
        Conv2D(numFilters[0], size_filt1, padding='same', name='conv1_1'),
        LeakyReLU(name='leakyReLU1_1', alpha=alpha),
        Dropout(dropout, name='dropout1_1'),
        
        Conv2D(numFilters[0], size_filt1, padding='same', name='conv1_2'),
        LeakyReLU(name='leakyReLU1_2', alpha=alpha),
        Dropout(dropout, name='dropout1_2'),
        
        Conv2D(numFilters[0], size_filt1, padding='same', name='conv1_3'),
        LeakyReLU(name='leakyReLU1_3', alpha=alpha),
        Dropout(dropout, name='dropout1_3'),
        
        Conv2D(numFilters[0], size_filt1, padding='same', name='conv1_4'),
        LeakyReLU(name='leakyReLU1_4', alpha=alpha),        
        MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid"),
        Dropout(dropout, name='dropout1'),
        
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        LeakyReLU(name='leakyReLU2', alpha=alpha),
        MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid"),
        Dropout(dropout, name='dropout2'),
        Conv2D(numFilters[2], (1,int(L/size2_filt2/4)), name='conv3'),
        LeakyReLU(name='leakyReLU3', alpha=alpha),
        Dropout(dropout, name='dropout3'),
        Conv2D(numFilters[3], (1,1), name='conv4'),
        LeakyReLU(name='leakyReLU4', alpha=alpha),
        Dropout(dropout, name='dropout4'),
        
        Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid'),
        Flatten(name='flatten')
        #Dense(12, activation='sigmoid', name='dense')
    ])    
    return cnn

# deep ResNet CNN (prefiltering layer is replicated + skip connections)
def musicalCNN_Jo_lastConv_maxPool_LN_deepResNet(B, C, L, numOctaves, numFilters=[20,20,10,1], size_filt1=(15,15), size2_filt2=3, stride2_filt2=3, dropout=0.2, alpha=0.3):

    inputs = Input(shape=(numOctaves*12*B, L, C))
    x = LayerNormalization(axis=[1,2,3])(inputs)
    #x = Conv2D(numFilters[0], (1,1), padding='same')(x)
    
    x = Conv2D(numFilters[0], size_filt1, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    
    
    for i in range(4):
        x = Dropout(dropout)(x)
        x_ = Conv2D(numFilters[0], size_filt1, padding='same')(x)
        x = tf.keras.layers.add([x, x_])
        x = LeakyReLU(alpha=alpha)(x)
        #x = Dropout(dropout)(x)
        
    #x_ = Conv2D(numFilters[0], size_filt1, padding='same')(x)
    #x = tf.keras.layers.add([x, x_])
    #x = LeakyReLU(alpha=alpha)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid")(x)
    x = Dropout(dropout)(x)
        
    x = Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2')(x)
    x = LeakyReLU(name='leakyReLU2', alpha=alpha)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid")(x)
    x = Dropout(dropout, name='dropout2')(x)
    x = Conv2D(numFilters[2], (1,int(L/size2_filt2/4)), name='conv3')(x)
    x = LeakyReLU(name='leakyReLU3', alpha=alpha)(x)
    x = Dropout(dropout, name='dropout3')(x)
    x = Conv2D(numFilters[3], (1,1), name='conv4')(x)
    x = LeakyReLU(name='leakyReLU4', alpha=alpha)(x)
    x = Dropout(dropout, name='dropout4')(x)

    x = Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid')(x)
    x = Flatten(name='flatten')(x)   
    return tf.keras.Model(inputs, x)

# 'inception' CNN (filters with different size in prefiltering layer)
def musicalCNN_Jo_lastConv_maxPool_LN_inception(B, C, L, numOctaves, configLayer1=[[5,(3,3)], [5,(9,9)], [5,(15,15)], [5,(27,27)]], numFilt_filt2=20, size2_filt2=3, stride2_filt2=3, numFilt_filt3=10, numFilt_filt4=1, dropout=0.2, alpha=0.3):
    
    inputs = Input(shape=(numOctaves*12*B, L, C))
    
    inputs_ = LayerNormalization(axis=[1,2,3])(inputs)
    
    filters = []
    for numFilt, sizeFilt in configLayer1:
        x = Conv2D(numFilt, sizeFilt, padding='same')(inputs_)
        filters.append(x)    
    filt1_out = tf.keras.layers.concatenate(filters, axis=-1)
    
    x = LeakyReLU(alpha=alpha)(filt1_out)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid")(x)
    x = Dropout(dropout)(x)

    x = Conv2D(numFilt_filt2, (B,size2_filt2), strides=(B, stride2_filt2))(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid")(x)
    x = Dropout(dropout)(x)

    x = Conv2D(numFilt_filt3, (1, int(L/size2_filt2/4)))(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dropout(dropout)(x)

    x = Conv2D(numFilt_filt4, (1,1), name='conv4')(x)
    x = LeakyReLU(name='leakyReLU4', alpha=alpha)(x)
    x = Dropout(dropout, name='dropout4')(x)

    x = Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid')(x)
    outputs = Flatten(name='flatten')(x)
    
    return tf.keras.Model(inputs, outputs)

# musically motivated CNN from Tim Zunners' Major Project
def musicalCNN_Tim(B, C, L, M, numOctaves):
    mCNN = Sequential([
            Conv2D(1, (M*B, L), strides=(B,L), padding='same', input_shape=(numOctaves*12*B, L, C), name='conv1'),
            ReLU(name='ReLU'),
            Conv2D(1, (numOctaves*12-11,1), padding='valid', activation='sigmoid', name='conv2'),
            Flatten(name='flatten')
            ])
    return mCNN



##############################################
############       LEGACY         ############

def musicalCNN_Jo1(B, C, L, numOctaves, numFilters=[10,10,5,1], size_filt1=(5,5), size2_filt2=3, stride2_filt2=3, dropout=0.2):
    cnn = Sequential([
        Conv2D(numFilters[0], size_filt1, padding='same',
            input_shape=(numOctaves*12*B, L, C), name='conv1'),
        LeakyReLU(name='leakyReLU1'),
        Dropout(dropout, name='dropout1'),
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        LeakyReLU(name='leakyReLU2'),
        Dropout(dropout, name='dropout2'),
        Conv2D(numFilters[2], (1,int(L/size2_filt2)), name='conv3'),
        LeakyReLU(name='leakyReLU3'),
        Dropout(dropout, name='dropout3'),
        Conv2D(numFilters[3], (1,1), name='conv4'),
        LeakyReLU(name='leakyReLU4'),
        Dropout(dropout, name='dropout4'),
        Flatten(name='flatten'),
        Dense(12, activation='sigmoid', name='dense')
    ])    
    return cnn

def musicalCNN_Jo_lastConv_maxPool(B, C, L, numOctaves, numFilters=[10,10,5,1], size_filt1=(5,5), size2_filt2=3, stride2_filt2=3, dropout=0.2, alpha=0.3):
    cnn = Sequential([
        Conv2D(numFilters[0], size_filt1, padding='same',
            input_shape=(numOctaves*12*B, L, C), name='conv1'),
        LeakyReLU(name='leakyReLU1', alpha=alpha),
        MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid"),
        Dropout(dropout, name='dropout1'),
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        LeakyReLU(name='leakyReLU2', alpha=alpha),
        MaxPooling2D(pool_size=(1, 2), strides=(1,2), padding="valid"),
        Dropout(dropout, name='dropout2'),
        Conv2D(numFilters[2], (1,int(L/size2_filt2/4)), name='conv3'),
        LeakyReLU(name='leakyReLU3', alpha=alpha),
        Dropout(dropout, name='dropout3'),
        Conv2D(numFilters[3], (1,1), name='conv4'),
        LeakyReLU(name='leakyReLU4', alpha=alpha),
        Dropout(dropout, name='dropout4'),
        
        Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid'),
        Flatten(name='flatten')
        #Dense(12, activation='sigmoid', name='dense')
    ])    
    return cnn

def musicalCNN_Jo_lastConv_ReLU(B, C, L, numOctaves, numFilters=[10,10,5,1], size_filt1=(5,5), size2_filt2=3, stride2_filt2=3, dropout=0.2):
    cnn = Sequential([
        Conv2D(numFilters[0], size_filt1, padding='same',
            input_shape=(numOctaves*12*B, L, C), name='conv1'),
        ReLU(),
        Dropout(dropout, name='dropout1'),
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        ReLU(),
        Dropout(dropout, name='dropout2'),
        Conv2D(numFilters[2], (1,int(L/size2_filt2)), name='conv3'),
        ReLU(),
        Dropout(dropout, name='dropout3'),
        Conv2D(numFilters[3], (1,1), name='conv4'),
        ReLU(),
        Dropout(dropout, name='dropout4'),
        
        Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid'),
        Flatten(name='flatten')
        #Dense(12, activation='sigmoid', name='dense')
    ])    
    return cnn

def musicalCNN_Jo_lastConv(B, C, L, numOctaves, numFilters=[10,10,5,1], size_filt1=(5,5), size2_filt2=3, stride2_filt2=3, dropout=0.2, alpha=0.3):
    cnn = Sequential([
        Conv2D(numFilters[0], size_filt1, padding='same',
            input_shape=(numOctaves*12*B, L, C), name='conv1'),
        LeakyReLU(name='leakyReLU1', alpha=alpha),
        Dropout(dropout, name='dropout1'),
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        LeakyReLU(name='leakyReLU2', alpha=alpha),
        Dropout(dropout, name='dropout2'),
        Conv2D(numFilters[2], (1,int(L/size2_filt2)), name='conv3'),
        LeakyReLU(name='leakyReLU3', alpha=alpha),
        Dropout(dropout, name='dropout3'),
        Conv2D(numFilters[3], (1,1), name='conv4'),
        LeakyReLU(name='leakyReLU4', alpha=alpha),
        Dropout(dropout, name='dropout4'),
        
        Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid'),
        Flatten(name='flatten')
        #Dense(12, activation='sigmoid', name='dense')
    ])    
    return cnn

def musicalCNN_Jo_lastConv_BN(B, C, L, numOctaves, numFilters=[10,10,5,1], size_filt1=(5,5), size2_filt2=3, stride2_filt2=3, dropout=0.2, alpha=0.3):
    cnn = Sequential([
        BatchNormalization(input_shape=(numOctaves*12*B, L, C)),
        Conv2D(numFilters[0], size_filt1, padding='same',
             name='conv1'),
        LeakyReLU(name='leakyReLU1', alpha=alpha),
        Dropout(dropout, name='dropout1'),
        Conv2D(numFilters[1], (B, size2_filt2), strides=(B, stride2_filt2), name='conv2'),
        LeakyReLU(name='leakyReLU2', alpha=alpha),
        Dropout(dropout, name='dropout2'),
        Conv2D(numFilters[2], (1,int(L/size2_filt2)), name='conv3'),
        LeakyReLU(name='leakyReLU3', alpha=alpha),
        Dropout(dropout, name='dropout3'),
        Conv2D(numFilters[3], (1,1), name='conv4'),
        LeakyReLU(name='leakyReLU4', alpha=alpha),
        Dropout(dropout, name='dropout4'),
        
        Conv2D(1, (numOctaves*12-11, 1), activation='sigmoid'),
        Flatten(name='flatten')
        #Dense(12, activation='sigmoid', name='dense')
    ])    
    return cnn

def octaveWiseCNN(B, C, L, numOctaves, numFilters=[10,10,5,1], size_filt1=(5,5), size2_filt2=3, stride2_filt2=3, dropout=0.2):
    inputs = Input(shape=(numOctaves*12*B, L, C))
    
    octaveFilters = []
    for o in range(numOctaves):
        x = Conv2D(numFilters[0], size_filt1, padding='same')(inputs[:,o*12*B:(o+1)*12*B,:,:])
        x = LeakyReLU()(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(numFilters[1], (B,size2_filt2), strides=(B, stride2_filt2))(x)
        x = LeakyReLU()(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(numFilters[2], (1, int(L/size2_filt2)))(x)
        x = LeakyReLU()(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(numFilters[3], (1,1))(x)
        
        octaveFilters.append(Flatten()(x) )
        
    octaveOutput = tf.keras.layers.concatenate(octaveFilters)
    outputs = Dense(12, activation='sigmoid')(octaveOutput)
    
    return tf.keras.Model(inputs, outputs)


def musicalCNN_distributedFirstLayer(B, C, L, numOctaves, configLayer1=[[5,(3,3)], [5,(9,9)], [5,(15,15)], [5,(27,27)]], numFilt_filt2=10, size2_filt2=3, stride2_filt2=3, numFilt_filt3=5, dropout=0.2):
    inputs = Input(shape=(numOctaves*12*B, L, C))
    
    filters = []
    for numFilt, sizeFilt in configLayer1:
        x = Conv2D(numFilt, sizeFilt, padding='same')(inputs)
        filters.append(x)    
    filt1_out = tf.keras.layers.concatenate(filters, axis=-1)
    
    x = LeakyReLU()(filt1_out)
    x = Dropout(dropout)(x)

    x = Conv2D(numFilt_filt2, (B,size2_filt2), strides=(B, stride2_filt2))(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    x = Conv2D(numFilt_filt3, (1, int(L/size2_filt2)))(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    x = Conv2D(1, (1,1))(x)

    x = Flatten()(x)
    
    outputs = Dense(12, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

def musicalCNN_distributedFilters_resNet(B, C, L, numOctaves, size_preFilt=(3,3), distFilters=[[5,(3,3)], [5,(9,9)], [5,(15,15)], [5,(27,27)]], numResBlocks=5, numFilt_filt2=10, size2_filt2=3, stride2_filt2=3, numFilt_filt3=5):
    
    numFilt_filt1 = 0
    for nf, _ in distFilters:
        numFilt_filt1 += nf
    
    inputs = Input(shape=(numOctaves*12*B, L, C))
    
    x = Conv2D(numFilt_filt1, size_preFilt, padding='same')(inputs)
    x = ReLU()(x)
    
    for i in range(numResBlocks):
        filters = []
        for numFilt, sizeFilt in distFilters:
            y = Conv2D(numFilt, sizeFilt, padding='same')(x)
            filters.append(y)
        filtOut = tf.keras.layers.concatenate(filters, axis=-1)
        filtOut = BatchNormalization()(filtOut)
        
        x = tf.keras.layers.add([x, filtOut])
        x = ReLU()(x)
        
    x = Conv2D(numFilt_filt2, (B, size2_filt2), strides=(B, stride2_filt2))(x)
    x = ReLU()(x)
    
    x = Conv2D(numFilt_filt3, (1, int(L/size2_filt2)))(x)
    x = ReLU()(x)
    
    x = Conv2D(1, (1,1))(x)

    x = Flatten()(x)
    
    outputs = Dense(12, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)
                                         
    
    