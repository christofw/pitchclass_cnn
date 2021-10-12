######################################################
#                   HARMONIC CQT                     #
#                                                    #
# Efficient implementation for harmonic CQT          #
#                                                    #
# Johannes Zeitler (johannes.zeitler@fau.de)         #
# Dec. 2020                                          #
######################################################

import numpy as np
import librosa

def HCQT(x, fs, frameRate=10, f_min='C1', binsPerKey=1, numOctaves=5, numHarmonics=1, centerNoteToBins=True, correctTuning=0.0, hopSizeCQT=None):
    #==========================================================================#
    # EFFICIENT HARMONIC CONSTANT Q TRANSFORM                                  #
    #                                                                          #
    # Idea: a single cqt can never exactly contain all harmonics. The cqt with #
    #  base f0 can contain harmonics 1, 2, 4, 8,... the cqt with base 3*f0 can #
    #  contain harmonics 3, 6, 12,...                                          #
    #                                                                          #
    # Example: 6 harmonics                                                     #
    # Necessary CQTS: f0 (1,2,4), 3*f0 (3,6), 5*f0 (5)                         #
    #========================= INPUTS =========================================#
    # x .................. input signal                                        #
    # fs ................. sampling frequency of input signal [Hz]             #
    # frameRate .......... frame rate for HCQT [Hz]                            #
    # f_min .............. lowest CQT frequency [Hz] or lowest note [string]   #
    # binsPerKey ......... CQT bins per key                                    #
    # numOctaves ......... number of CQT octaves                               #
    # numHarmonics ....... number of harmonics for HCQT                        #
    # centerNoteToBins ... if f_min is a note [string] and binsPerKey >=3,     #
    #  whether to center the note frequency at binsPerKey                      #
    #                                                                          #
    #========================= OUTPUT =========================================#
    # HCQT ............... output HCQT [cqt bins x time frames x harmonics]    #
    # hopSizeCQT ......... CQT hop size                                        #
    # cqtTimes ........... CQT time steps                                      #
    #==========================================================================#
    
    if isinstance(correctTuning, bool):
        if correctTuning:
            tuning = librosa.estimate_tuning(x,fs, bins_per_octave=12)*binsPerKey
        else:
            tuning = 0
    else:
        tuning = float(correctTuning)
        
    #print('Tuning: %.3f'%(tuning))

    
    eps = 1e-15
    binsPerOctave = binsPerKey*12    
    
    # max. number of octaves to be processed in a single cqt
    maxCQT_octaves = np.floor(np.log2(numHarmonics)) + numOctaves
    
    if hopSizeCQT is None:
        # CQT hop size is integer multiple of 2**maxCQT_octaves
        hopSizeCQT = getHopSize(fs, frameRate, numOctaves, numHarmonics)
    
    
    # if f_min is a string, convert note to hz
    if isinstance(f_min, str):
        f_min = librosa.note_to_hz(f_min)
        
        # if true, center the note frequencies. I.e. if binsPerKey=3, the piano note frequencies are on bins 2, 5, 8,...
        if centerNoteToBins:
            f_min = f_min / (2**((binsPerKey-1)/2/binsPerOctave))
            
    
    
    # for all harmonics, find the 'basic' cqt where they can be derived from
    from_cqt = np.zeros((numHarmonics), dtype='int')    
    for h in range(1, numHarmonics+1):
        # check if harmonic is odd -> we need a new cqt here
        if np.mod(h,2) > eps:
            from_cqt[h-1] = h
        # if harmonic is even, get the lowest multiple of f0 that can be used as a base cqt
        else:
            base = h
            while np.mod(base, 2) < eps:
                base = base/2
            from_cqt[h-1] = base
    
    addFrame = 0
    if np.mod(len(x), hopSizeCQT) < eps:
        addFrame = 1
        
    HCQT = np.zeros((numOctaves*binsPerOctave, np.ceil(len(x)/hopSizeCQT).astype(int)+addFrame, numHarmonics))#, dtype=complex)
    
    # harmonics that have already been processed
    processedHarmonics = []
    
    # development purposes
    additionalOctavesCntr = 0
    for h in range(1, numHarmonics+1):
        #print('h: %i'%(h))
        if h in processedHarmonics:
            #print('continue')
            continue
        numHarmonicsForCQT = np.sum(from_cqt==h)
        additionalOctaves = numHarmonicsForCQT - 1
        additionalOctavesCntr += additionalOctaves
        
        #print('Perform cqt with fmin=%.8f, nBins=%i and hopLength=%i'%(f_min*h, int((numOctaves+additionalOctaves)*binsPerOctave), hopSizeCQT))
        # perform the cqt
        cqt = librosa.cqt(x, sr=fs, hop_length=hopSizeCQT, 
                          fmin=f_min*h, 
                          n_bins=int((numOctaves+additionalOctaves)*binsPerOctave), 
                          bins_per_octave=binsPerOctave,
                          tuning = tuning)
        
        # extract harmonics from cqt and insert in HCQT
        for h_ in range(1, numHarmonics+1):
            #print('h_: %i'%(h_))
            if from_cqt[h_ - 1] == h:
                cqtOctave = np.log2(h_/h).astype(int)
                #print('Insert harmonic %i to HCQT, cqt bins from %i to %i'%(h_, cqtOctave*binsPerOctave, (cqtOctave+numOctaves)*binsPerOctave))
                
                #print('CQT shape: ',cqt.shape)
                HCQT[:,:,h_ - 1] = np.abs(cqt[cqtOctave*binsPerOctave:(cqtOctave+numOctaves)*binsPerOctave,:])
                processedHarmonics.append(h_)
                
    cqtTimes = librosa.times_like(HCQT.shape[1], sr=fs, hop_length=hopSizeCQT)
    
    return HCQT, hopSizeCQT, cqtTimes

def subHCQT(x, fs, frameRate=10, f_min='C1', binsPerKey=1, numOctaves=5, numHarmonics=1, centerNoteToBins=True, correctTuning=0.0, hopSizeCQT=None, numSubHarmonics=1):
    
    if isinstance(correctTuning, bool):
        if correctTuning:
            tuning = librosa.estimate_tuning(x,fs, bins_per_octave=binsPerKey*12)
        else:
            tuning = 0
    else:
        tuning = float(correctTuning)
        
    print('Tuning: %.3f'%(tuning))

    
    eps = 1e-15
    binsPerOctave = binsPerKey*12    
    
    # max. number of octaves to be processed in a single cqt
    maxCQT_octaves = np.floor(np.log2(numHarmonics)) + numOctaves
    
    if hopSizeCQT is None:
        # CQT hop size is integer multiple of 2**maxCQT_octaves
        hopSizeCQT = getHopSize(fs, frameRate, numOctaves, numHarmonics)
    
    
    # if f_min is a string, convert note to hz
    if isinstance(f_min, str):
        f_min = librosa.note_to_hz(f_min)
        
        # if true, center the note frequencies. I.e. if binsPerKey=3, the piano note frequencies are on bins 2, 5, 8,...
        if centerNoteToBins:
            f_min = f_min / (2**((binsPerKey-1)/2/binsPerOctave))
            
    hcqt,_,cqtTimes = HCQT(x, fs, frameRate=frameRate, f_min=f_min, binsPerKey=binsPerKey, numOctaves=numOctaves, numHarmonics=numHarmonics, correctTuning=tuning, hopSizeCQT=hopSizeCQT)
    
    for s in range(numSubHarmonics):
        fSub = f_min/(s+1)
        shcqt, _, _ = HCQT(x, fs, frameRate=frameRate, f_min=fSub, binsPerKey=binsPerKey, numOctaves=numOctaves, numHarmonics=1, correctTuning=tuning, hopSizeCQT=hopSizeCQT)
        
        hcqt = np.concatenate((shcqt, hcqt), axis=2)
    
    return hcqt, hopSizeCQT, cqtTimes

    
    
def getHopSize(fs, frameRate, numOctaves, numHarmonics):
    maxCQT_octaves = np.floor(np.log2(numHarmonics)) + numOctaves
    
    # CQT hop size is integer multiple of 2**maxCQT_octaves
    hopSizeCQT = int(round(fs/frameRate/(2**(maxCQT_octaves-1)))*2**(maxCQT_octaves-1))
    
    return hopSizeCQT
    