# pitchclass_cnn

This is a Keras code repository accompanying the following paper:  

> Christof Weiß, Johannes Zeitler, Tim Zunner, Florian Schuberth, and Meinard Müller
> _Learning Pitch-Class Representations from Score-Audio Pairs of Classical Music_  
>  Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2021  

&copy; Johannes Zeitler (johannes.zeitler@fau.de) and Christof Weiß (christof.weiss@audiolabs-erlangen.de), 2020/21

This repository only contains exemplary code and pre-trained models for most of the paper's experiments as well as some individual examples. Some of the datasets used in the paper are publicly available (at least partially), e.g.:
* [Schubert Winterreise Dataset (SWD)](https://zenodo.org/record/5139893#.YWRcktpBxaQ)
* [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)
For details and references, please see the paper.


## 1. Overview: files and folders

### Folders:
* _Data:_    Exemplary data folder with some examples from Schubert Winterreise Dataset
* _LibFMP:_  AudioLabs LibFMP library, see [here](https://pypi.org/project/libfmp/) for an update version
* _Models:_  Pre-trained CNN models
* _PrecomputedResults_: Pre-computed evaluation measures, to be loaded for reproducing the figures in the paper
* _TrainScripts_:       Python files for automated model training - __just for information, not executable due to missing data!__


### Jupyter Notebooks:
* _01_preprocess_data_schubert_winterreise.ipynb:_ Dataset preprocessing, demonstrated at Schubert's Winterreise
* _02_evaluate_model_parameters.ipynb:_    Evaluate impact of basic model parameters (__first part of Section 4 in the paper__)
* _03_evaluate_datasets_and_models.ipynb:_ Training/testing on different datasets and with different networks (__Figures 3 and 4 in the paper__)
* _04_demo_estimate_pitchclasses.ipynb:_   Load an audio file and estimate pitch classes with a pre-trained CNN


### Python Scripts and libraries:
* _estimatePitchClasses.py:_ Command line tool to estimate chromagram with a pre-trained CNN
* _customModels:_    CNN model definitions
* _FrameGenerators:_ Tensorflow generators for feeding data to train, evaluate and predict functions
* _harmonicCQT:_     Efficient implementation of the harmonic constant-Q-transform (HCQT)
* _utils:_           Collection of useful functions for preprocessing etc.
* _utils_DL:_        Collection of useful functions for the deep-learning pipeline
 
### Environment file:
* _environment.yml_: To install Python/Keras environment _pitchclass_cnn_
 

## 2. Example: predict pitch classes with pre-trained models
Start the file __estimatePitchClasses.py__ from a Python shell:  
__conda activate pitchclass_cnn__  
__python estimatePitchClasses.py -s <audio_file.wav> -t <target_file.npy> -r__
__<sample rate of output features> -n (L2-normalize feature sequence)__
