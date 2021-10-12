######################################################
#               UTILITY FUNCTIONS                    #
#                                                    #
# Some utility functions that were written while     #
#  working on deep chroma estimation                 #
#                                                    #
# Johannes Zeitler (johannes.zeitler@fau.de)         #
# Dec. 2020                                          #
######################################################

import numpy as np
import matplotlib.pyplot as plt
import librosa
import pretty_midi
import os
import LibFMP.B
import IPython.display as ipd

# split long audio files into shorter segments of a defined duration
def splitLargeFiles(files, cqt_dir, chroma_dir, frameRate=50, maxDuration_seconds=100, suffix='_short', saveAlways=False):
    maxFrames = int(maxDuration_seconds*frameRate)
    for f in files:
        cqt = np.load(os.path.join(cqt_dir, f))
        chroma = np.load(os.path.join(chroma_dir, f))
        fileLen = chroma.shape[1]
        if fileLen < maxFrames:
            if saveAlways:
                fname, ftype = f.split('.')
                np.save(os.path.join(cqt_dir, fname+suffix+str(0)+'.'+ftype), cqt)
                np.save(os.path.join(chroma_dir, fname+suffix+str(0)+'.'+ftype), cqt)
            continue
        else:
            splitInds=np.arange(maxFrames, fileLen, maxFrames)

            cqtSplits=np.split(cqt, splitInds, axis=1)
            chromaSplits = np.split(chroma, splitInds, axis=1)

            fname, ftype = f.split('.')
            for i in range(len(cqtSplits)):
                np.save(os.path.join(cqt_dir, fname+suffix+str(i)+'.'+ftype), cqtSplits[i])
                np.save(os.path.join(chroma_dir, fname+suffix+str(i)+'.'+ftype), chromaSplits[i])

# sonify pitch activations and audio signal. Adds possibility to specify start and end time
def sonifyStereo(pitch, x, hopSize, fs, min_pitch, limTime=None, harmonics_weights=[1, .5]):
    if limTime is not None:
        x = x[np.max((0,int(limTime[0]*fs))) : np.min((len(x), int(limTime[1]*fs)))]
        pitch = pitch[:, np.max((0, int(limTime[0]*fs/hopSize))):np.min((pitch.shape[1], int(limTime[1]*fs/hopSize)))]

    _, out = LibFMP.B.sonify_pitch_activations_with_signal(pitch, x, fs/hopSize, fs, min_pitch=min_pitch, Fc=440, harmonics_weights=harmonics_weights, fading_msec=5, stereo=True)
    return out

# sonify pitch activations. Adds possibility to specify start and end time
def sonifyMONO(pitch, hopSize, fs, min_pitch=60, limTime=None, harmonics_weights=[1, .5], chroma=False):
    if limTime is not None:
        pitch = pitch[:, np.max((0, int(limTime[0]*fs/hopSize))):np.min((pitch.shape[1], int(limTime[1]*fs/hopSize)))]

    if chroma:
        out = LibFMP.B.sonify_chromagram(pitch, int(pitch.shape[1]*hopSize),
                                         fs/hopSize, fs, fading_msec=5)
    else:
        out = LibFMP.B.sonify_pitch_activations(pitch, int(pitch.shape[1]*hopSize), fs/hopSize, fs, min_pitch=min_pitch, Fc=440, harmonics_weights=harmonics_weights, fading_msec=5)
    return out


# check if audio should be shifted to match a cqt better
# output: number of frames that the pitch activations should be shifted to match the audio (negative = to the left)
def checkFrameShift(cqt, pitch, maxShift=20, numFrames=None):
    shiftInds = np.arange(-maxShift, maxShift+1, 1)
    startFrame = maxShift

    if numFrames is None:
        endFrame = cqt.shape[1] - maxShift - 1
    else:
        endFrame = np.min((startFrame+numFrames, cqt.shape[1]-maxShift-1))

    cqtSnip = cqt[:, startFrame:endFrame]

    minInd = np.argmin([np.sum(cqtSnip*(1-pitch[:,startFrame-i:endFrame-i])) for i in shiftInds])
    return shiftInds[minInd]

# shift pitch and chroma activations by specified number of frames.
# negative shift = to the left
def shiftFrames(pitch, chroma, shift):

    if shift == 0:
        return pitch, chroma

    pitch_ = np.zeros_like(pitch)
    chroma_ = np.zeros_like(chroma)

    if shift > 0:
        pitch_[:,shift:] = pitch[:,:-shift]
        chroma_[:,shift:] = chroma[:,:-shift]

    else:
        pitch_[:,:shift] = pitch[:,-shift:]
        chroma_[:,:shift] = chroma[:,-shift:]

    return pitch_, chroma_

# check if pitch and chroma activations should be shifted to be synchronized to audio data, then shift accordingly
def checkAndShift(cqt, pitch, chroma, maxShift=20, numFrames=None, isCenteredCQT=False):
    if isCenteredCQT:
        binsPerKey = int(cqt.shape[0]/pitch.shape[0])
        startInd = int((binsPerKey-1)/2)
        cqt = cqt[np.arange(startInd, cqt.shape[0], binsPerKey),:]

    if len(cqt.shape)>2:
        cqt = cqt[:,:,0]
    shift = checkFrameShift(cqt, pitch, maxShift, numFrames)
    pitch, chroma = shiftFrames(pitch, chroma, shift)
    return pitch, chroma, shift

# transform a list of note events (from csv or similar) to pitch and chroma activation matrices. Captures every note event.
def score_to_matrices(score, timeSteps, bottomPitch=0, topPitch=127, field_t0=0, field_t1=1, field_pitch=2, controlChanges=None):
    ''' OLD VERSION, doesn't recognize fast staccato
    pitchMatrix = np.zeros((topPitch-bottomPitch+1, len(timeSteps)))
    chromaMatrix = np.zeros((12, len(timeSteps)))

    for i in range(score.shape[0]):
        t0 = score[i, field_t0]
        t1 = score[i, field_t1]
        p = score[i, field_pitch]

        chromaMatrix[int(np.mod(p-60,12)),:] += ((t0 < timeSteps)*1) * ((t1>timeSteps)*1)

        if p < bottomPitch or p > topPitch:
            continue

        pitchMatrix[int(p-bottomPitch),:] += ((t0 < timeSteps)*1) * ((t1>timeSteps)*1)

    return  (pitchMatrix > 0)*1.0, (chromaMatrix > 0)*1.0
    '''

    pitchMatrix = np.zeros((topPitch-bottomPitch+1, len(timeSteps)))
    chromaMatrix = np.zeros((12, len(timeSteps)))

    if isinstance(score, np.ndarray):
        score = [score[i,:] for i in range(score.shape[0])]

    for i in range(len(score)):
        t0 = score[i][field_t0]
        t1 = score[i][field_t1]
        p = score[i][field_pitch]

        if p < bottomPitch or p > topPitch:
            continue

        fAct = np.arange(len(timeSteps))[((t0 < timeSteps) * (t1>timeSteps)).astype(bool)]

        if len(fAct) < 1:
            startFrame = np.argmin(np.abs(timeSteps-t0))
            stopFrame = np.argmin(np.abs(timeSteps-t1))

            if np.abs(timeSteps[startFrame] - t0) < np.abs(timeSteps[stopFrame] - t1):
                fAct = startFrame
            else:
                fAct = stopFrame

        pitchMatrix[int(p-bottomPitch), fAct] = 1

    # process sustain pedal like pretty midi does
    if controlChanges is not None:
        fs = 1 / np.mean(timeSteps[1:]-timeSteps[:-1])
        pedal_threshold = 64
        CC_SUSTAIN_PEDAL = 64
        time_pedal_on = 0
        is_pedal_on = False
        for cc in [_e for _e in controlChanges
                   if _e.number == CC_SUSTAIN_PEDAL]:
            time_now = int(cc.time*fs)
            is_current_pedal_on = (cc.value >= pedal_threshold)
            if not is_pedal_on and is_current_pedal_on:
                time_pedal_on = time_now
                is_pedal_on = True
            elif is_pedal_on and not is_current_pedal_on:
                # For each pitch, a sustain pedal "retains"
                # the maximum velocity up to now due to
                # logarithmic nature of human loudness perception
                subpr = pitchMatrix[:, time_pedal_on:time_now]

                # Take the running maximum
                pedaled = np.maximum.accumulate(subpr, axis=1)
                pitchMatrix[:, time_pedal_on:time_now] = pedaled
                is_pedal_on = False

    for p in range(bottomPitch, topPitch+1):
        chromaMatrix[int(np.mod(p,12)),:] += pitchMatrix[p-bottomPitch,:]

    return (pitchMatrix>0)*1.0, (chromaMatrix>0)*1.0



# transform a midi file to pitch and chroma activation matrices
def midi_to_matrices(midi, timeSteps, bottomPitch=0, topPitch=127):

    ''' OLD VERSION
    midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    pnoRoll = np.zeros((topPitch-bottomPitch+1, len(timeSteps)))
    chroma = np.zeros((12, len(timeSteps)))

    for instrument in midi_data.instruments:
        roll = instrument.get_piano_roll(fs=fs/hopLength, times=timeSteps)
        pnoRoll += (roll[bottomPitch:topPitch+1,:] > 0)*1.0
        chroma += (instrument.get_chroma(fs=fs/hopLength, times=timeSteps) > 0)*1

    return (pnoRoll>0)*1.0, (chroma>0)*1.0
    '''

    pitchMatrix = np.zeros((topPitch-bottomPitch+1, len(timeSteps)))
    chromaMatrix = np.zeros((12, len(timeSteps)))

    midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        notes = []
        for n in instrument.notes:
            notes.append([n.start, n.end, n.pitch])

        CCs = instrument.control_changes

        pitch, chroma = score_to_matrices(notes, timeSteps, bottomPitch, topPitch, controlChanges=CCs)
        pitchMatrix += pitch
        chromaMatrix += chroma

    return (pitchMatrix>0)*1.0, (chromaMatrix>0)*1.0


# overlay cqt and pitch activation plot
def validate_alignments(cqt, pitch, hopLength, bottomNote='C1', fs=22050, binsPerKey=1, centerNoteToBins=True, compress=0, limTime=None, limPitch=None, ticksPerOctave=3, figsize=(15,10)):
    if compress:
        cqt = np.log10(1+compress*cqt)

    time = librosa.times_like(cqt, sr=fs, hop_length=hopLength)

    nBins = cqt.shape[0]
    nKeys = nBins/binsPerKey

    startBin = 0
    if centerNoteToBins:
        startBin += int((binsPerKey-1)/2)

    ind_bins = np.arange(0, nKeys, ticksPerOctave)
    ind_midi = librosa.midi_to_note(ind_bins+librosa.note_to_midi(bottomNote))

    pitch_alpha = np.zeros((cqt.shape[0], pitch.shape[1], 4))
    for i in range(pitch.shape[0]-1):
        cqtLine = i*binsPerKey + startBin
        pitch_alpha[cqtLine,:,0] = 1
        pitch_alpha[cqtLine,:,3] = pitch[i,:]*0.5


    plt.figure(figsize=figsize)
    plt.imshow(cqt[::-1,:], extent=(time[0], time[-1],-startBin/binsPerKey-1/(2*binsPerKey),nKeys-1+startBin/binsPerKey+1/(2*binsPerKey)), aspect='auto', cmap='gray_r')
    plt.imshow(pitch_alpha[::-1,:,:], extent=(time[0], time[-1], -startBin/binsPerKey-1/(2*binsPerKey),nKeys-1+startBin/binsPerKey+1/(2*binsPerKey)), aspect='auto')
    plt.yticks(ind_bins, ind_midi)
    if limTime is not None:
        plt.xlim([limTime[0], limTime[1]])
    if limPitch is not None:
        plt.ylim([librosa.note_to_midi(limPitch[0])-librosa.note_to_midi(bottomNote), librosa.note_to_midi(limPitch[1])-librosa.note_to_midi(bottomNote)])
    plt.grid()
    plt.xlabel('Time [s]')
    plt.show()


# investigate chromagram statistics for a specific dataset (i.e. percentage of active frames...)
def chromaStatistics(datasetPath, fileList=None, plot=False, frameRate=10):
    eps = 1e-15
    if fileList is None:
        fileList = os.listdir(datasetPath)

    numFrames = 0
    noChromaFrame = 0
    oneChromaFrame = 0
    chromas = np.zeros(12)

    for f in fileList:
        ch = np.load(os.path.join(datasetPath, f))
        numFrames += ch.shape[1]
        noChromaFrame += np.sum(np.sum(ch, axis=0) < eps)
        oneChromaFrame += np.sum(np.abs(np.sum(ch, axis=0) - 1) < eps)
        chromas += np.sum(ch, axis=1)

    duration = numFrames/frameRate

    print('Statistics for ',datasetPath)
    print('Number of frames: %i, approximately %.2f hours'%(numFrames, duration/3600))
    print('Number of frames without active chroma: %i = %.2f%%'%(noChromaFrame, noChromaFrame/numFrames*100))
    print('Number of frames with ONE active chroma: %i = %.2f%%'%(oneChromaFrame, oneChromaFrame/numFrames*100))

    if plot:
        plt.figure()
        plt.stem(chromas/numFrames, use_line_collection=True)
        xlabels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        plt.xticks(np.arange(12), xlabels)
        plt.xlabel('Chroma')
        plt.ylabel('Frequency')
        plt.title('Frequency of Chroma Values in '+datasetPath)
        plt.show()

    print('------------------------------------------\n')

    return {'dataset': datasetPath,
           'numFrames': numFrames,
           'noChromaFrame': noChromaFrame,
           'oneChromaFrame': oneChromaFrame,
           'chromas':chromas}
