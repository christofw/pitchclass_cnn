#######################################################################################
#                            DEEP PITCH CLASS ESTIMATOR                               #
#                                                                                     #
# Estimates binary pitch class activation (chroma) from audio data using a trained    #
#  convolutional neural network. Use <python estimatePitchClasses.py -h> for help.         #
#                                                                                     #
# Johannes Zeitler (johannes.zeitler@fau.de)                                          #
# Dec. 2020                                                                           #
#######################################################################################

import os
import sys
import getopt
import numpy as np
import librosa
from harmonicCQT import HCQT, getHopSize
import tensorflow as tf
from FrameGenerators import PredictGenerator
from LibFMP.C3 import smooth_downsample_feature_sequence, normalize_feature_sequence

def main(src, tgt, rate, normalize):

    if rate not in (10, 50):
        print('Error: output rate must be 10Hz or 50Hz')
        return

    fs = 22050 # audio sampling frequency

    # load audio
    print('load and resample audio...')
    audioIn, _ = librosa.load(src, sr=fs, mono=True)

    # HCQT config
    cqt_frameRate = 50 # desired cqt rate (actual cqt rate is approx. 57Hz due to power-of-two-constraints...)
    bottomNote = 'C1'
    bottomPitch = librosa.note_to_midi(bottomNote)
    numOctaves = 6
    numHarmonics = 5
    binsPerKey = 3
    hopSizeCQT = getHopSize(fs, cqt_frameRate, numOctaves, numHarmonics) # actual cqt hop size (corresponds to 57Hz)

    # compute HCQT
    print('compute hcqt...')
    hcqt, _, _ = HCQT(x=audioIn, fs=fs, frameRate=cqt_frameRate, f_min=bottomNote, binsPerKey=binsPerKey,
                      numOctaves=numOctaves, numHarmonics=numHarmonics, centerNoteToBins=True,
                      correctTuning=True)

    # tensorflow generator for feeding HCQT batches to the network. Decrease batch_size if insufficient memory.
    predictGen = PredictGenerator(hcqt, batch_size=100, numContextFrames=75, compression=10)


    #### Select model, load and compile the network ############################

    ## Models from the cross-dataset experiments (Figure 3 of the paper): ##
    # modelName = 'mCNN_trainMusicNet' # Trained on MusicNet
    # modelName = 'mCNN_trainBeethovenPiano' ' # Trained on Beethoven Piano Sonatas
    # modelName = 'mCNN_trainWagnerPianoScore' # Trained on Wagner Ring
    # modelName = 'mCNN_trainMusicNet-BeethovenPiano-WagnerPianoScore' # Trained on Mix

    ## Models from different architectures trained on a mixed dataset (Figure 4 of the paper): ##
    # modelName = 'mCNN_trainMusicNet-BeethovenPiano-WagnerPianoScore'  # Basic
    # modelName = 'mCNN_trainMusicNet-BeethovenPiano-WagnerPianoScore_LastLayer10' # BasicLast10
    # modelName = 'mCNN_WIDE_trainMusicNet-BeethovenPiano-WagnerPianoScore' # Wide
    # modelName = 'mCNN_WIDE-Inception_trainMusicNet-BeethovenPiano-WagnerPianoScore' # WideInception
    # modelName = 'mCNN_DEEP_trainMusicNet-BeethovenPiano-WagnerPianoScore' # Deep
    # modelName = 'mCNN_DEEP-ResNet_trainMusicNet-BeethovenPiano-WagnerPianoScore' # DeepResNet

    ## Pitch-class features for chord recognition (Section 5 of the paper) ##
    modelName = 'mCNN_WIDE_trainMusicNet-BeethovenPiano-WagnerPianoScore'    # for chord reco on Schubert Winterreise
    # modelName = 'mCNN_WIDE_trainMusicNet-SMD-WagnerPianoScore'    # for chord reco on Beethoven Sonatas

    ############################################################################


    print('load model ' + modelName)
    model = tf.keras.models.load_model(os.path.join('Models', modelName), compile=False)
    model.compile(loss = 'binary_crossentropy')

    # predict chromas
    print('predict chromas...')
    output = model.predict(predictGen).transpose()

    # interpolate chromas to be sampled at _exactly_ 50Hz (instead of 57Hz, see above)
    cqtTimes = np.arange(output.shape[1]) / (fs/hopSizeCQT)
    targetTimes = np.arange(0, np.max(cqtTimes), 1/cqt_frameRate)
    outputInterp = np.zeros((12, len(targetTimes)))
    for i in range(12):
        outputInterp[i,:] = np.interp(targetTimes, cqtTimes, output[i,:])

    # if required, downsample to 10Hz
    if rate is 10:
        outputInterp,_ = smooth_downsample_feature_sequence(outputInterp, fs, filt_len=5,
                                                                      down_sampling=5, w_type='boxcar')

    # frame-wise normalization
    if normalize:
        outputInterp = normalize_feature_sequence(outputInterp, norm='2', threshold=0.0001, v=None)

    # save chromagram
    np.save(tgt, outputInterp)

    return tgt

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hs:t:r:n',[])
    except getopt.GetoptError:
        print('Invalid usage')
        sys.exit()
    if not opts:
        print('Invalid usage')
        sys.exit()

    rate = 10
    normalize=False

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: estimatePitchClasses.py \n predict pitch class activation from audio files \n -s <source file> \n -t <target file> \n -r <sample rate of output features> default: 10 \n -n normalize feature sequence (l2-norm)')
            sys.exit()
        elif opt in ('-s'):
            src = arg
        elif opt in ('-t'):
            tgt = arg
        elif opt in ('-r'):
            rate = int(arg)
        elif opt in ('-n'):
            normalize=True

    main(src, tgt, rate, normalize)
