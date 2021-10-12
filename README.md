# pitchclass_cnn

This is a Keras code repository accompanying the following paper:  

> Christof Weiß, Johannes Zeitler, Tim Zunner, Florian Schuberth, and Meinard Müller
> _Learning Pitch-Class Representations from Score-Audio Pairs of Classical Music_  
>  Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2021  

&copy; Johannes Zeitler (johannes.zeitler@fau.de) and Christof Weiß (christof.weiss@audiolabs-erlangen.de), 2020/21


################################### FILES & FOLDERS ##########################################

Folders:

 Deprecated........... development code scripts that are no longer needed
 figures.............. some figures that were saved from scripts
 Figures_Report....... generates all plots for report
 LibFMP............... AudioLabs LibFMP library
 logs................. training log files
 Models............... trained CNN models
 Presentation_Dec20... code and figures for final presentation
 Presentation_Nov20... code and figures for intermediate presentation
 processData.......... jupyter notebooks for data preprocessing (copy back to main (Code) folder for execution)
 trainScripts......... python files for automated model training (copy back to main (Code) folder for execution)


Jupyter Notebooks:

 checkFrameShifts_SMD............... check and correct alignment of audio and labels for SMD 
 eval_InputComplexity............... evaluate impact of basic input and network complexity
 evaluateAndPredict_BeethovenPiano.. predict and evaluate chroma estimation for Beethoven
 ExploreDatasets.................... dataset statistics
 mCNN_otherDatasets_eval2........... training/testing on different datasets and with different networks
 mCNN_otherDatasets_eval_Piano...... training/testing on different piano datasets
 mCNN_Schubert_crossVal-eval........ evaluation of Schubert cross-validation 
 mCNN_Schubert_crossVal_HPRS_eval... evaluation of hrps-assisted training
 mCNN_Schubert_trainOnPitch......... pre-training on pitch annotations
 musicalCNN_chromas_sonifications... sonifications and detailed plots of estimated chromagrams
 numberOfParameters................. overview of number of trainable parameters in different networks
 processData_Schubert_Winterreise_tuning_50HZ
  .................................. Dataset preprocessing, demonstrated at Schubert's Winterreise
 splitDatasets...................... split large files into smaller pieces for training purposes
 trainingCurves..................... training metrics for different architectures and datasets
 verifyEfficientHCQT................ check whether efficient HCQT implementation matches the 'brute force' version


Python Scripts:

 estimateChromas........................ load an audio file and estimate chromagram with a pre-trained CNN
 mCNN_Schubert_crossVal_HPRS............ HRPS test script
 musicalCNN_Jo1_Schubert_TestScript..... initial grid search over network parameters
 musicalCNN_Jo_lastConv_maxPool_Schubert_50Hz_crossVal
  ...................................... cross-Validaiton on Schubert dataset with proposed model and different BCE weights
 musicalCNN_Tim_Beethoven_50Hz_crossVal. cross-validation with Tim's mCNN on Beethoven
 musicalCNN_Tim_Schubert_50Hz_crossVal.. cross-validation with Tim's mCNN on Schubert
 musicalCNN_Tim_Schubert_TestScript..... initial grid search over network parameters with Tim's mCNN


Libraries:

 customModels....... CNN model definitions
 FrameGenerators.... tensorflow generators for feeding data to train, evaluate and predict functions
 harmonicCQT........ efficient implementation of harmonic constant-Q-transform
 utils.............. collection of useful functions
 utils_DL........... collection of useful Deep-Learning-related functions
 
 
 
 
################################## BASIC WORKFLOW ##########################################

To get an overview of the basic workflow used in this projects, take a look at the following files
 
   Data preprocessing:
     processData_...    compute hcqts and ground-truth chromagrams
     splitDatasets      split songs into smaller segments for better training performance
   
   Training:
     trainScript_...   Model and training setup (choose e.g. trainScript_..._zerosWeight20 for custom BCE-loss)
       -> customModels: model Definition
       -> FrameGenerators: Train & Validation data generation
       -> utils_DL: autoTrain() function for automated training
       
   Evaluation:
     mCNN_otherDatasets_eval2: evaluate performance of a pre-trained model
   
   Prediction:
     estimateChromas.py: predict chromagrams from audio with pre-trained model