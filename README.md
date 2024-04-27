# Digital Signal Processing Project
Instituto Superior Técnico

Digital Signal Processing

Prof. Luis Caldas de Oliveira

April 2023

## Authors 

Group 03

Yandi Jiang, 96344

Bruno Pedro, 96363

## Instructions and Description

This repository concerns the final version of the project of Digital Signal Processing (DSP) course, at Instituto Superior Técnico, University of Lisbon. Its main objetive is the extraction of features from audio recordings of instruments interpreting musical melodies and its reinterpretation by a digital polyphonic synthesizer.

The features to be extracted include the detection of note onsets, the pitch, amplitude and duration of each note and, also, the beat and tempo of the melody. The features extracted are, then, organized and used to synthesize a new audio with a different timbre, representing a new interpretation of the same musical melody. The synthesized audio can be modified in order to represent different instruments, and, also, to change some characteristics of the original melody, such as the beat and tempo, the ADSR (attack, decay, sustain, release) envelope of the sound waveform, the octave of the notes or, finally, the addition of an overlaid delayed sound.

The notebook of this project is organized in three main parts:

* The first part includes the actual code functions developed, 

* The second part is where the developed code is applied to some audio recordings and the performance of the obtained results is assessed. First, concerning the onset detection, the performance of the implemented algorithms is evaluated by comparison with human annotations of the true onsets and the results are presented in tables that show the number of True Positives, False Positives and False Negatives onsets, and Precision, Recall and F-measure values. The avarage F-measure for each onset detection method is, also, computed and the results are shown and commented. Having the best onset detector selected for each audio, the features are, finally, extracted from the audios, namely, the onsets, pitch, amplitude and duration of notes, and beat of the melody. Then, the developed polyphonic synthesizer and audio modification features are applied, in detail, to a specific audio recording of piano ('silhuette.wav') Finally, some final results to demonstrate all the capabilities of the code are presented using different audios of viola and piano. In this case, a qualitative evaluation of the results is done, since it is hard to quantify the performance of synthesized audio.

* The third part addresses some simple code tests of the main functions developed, namely, the onset detection, extraction of pitch, amplitude and duration of notes and the capacity of polyphony synthesis. For the code tests, some audios with well-known characteristics are used.

## Video of the Project

https://www.youtube.com/embed/hHfP0OxZERU
