#!pip install madmom
import numpy as np
import librosa
from tabulate import tabulate
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from IPython.display import Audio
from matplotlib import pyplot as plt
from madmom.audio.signal import Signal
from madmom.audio.stft import ShortTimeFourierTransform
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogram
from madmom.audio.spectrogram import Spectrogram # Return the Spectrogram Class

from madmom.features.onsets import CNNOnsetProcessor
from madmom.audio.signal import FramedSignal
from madmom.features.onsets import RNNOnsetProcessor
from madmom.features.onsets import superflux
from madmom.features.onsets import complex_flux
from madmom.features.onsets import OnsetPeakPickingProcessor
from madmom.evaluation.onsets import OnsetEvaluation

def find_threshold(Signal, onset_processor, Human_onset, limit, fps, window_size):
    threshold_list = []
    best_f1score = 0
    for signal,human_onset in zip(Signal, Human_onset):
        print(signal)
        if(not isinstance(Signal, list)): # if List only has one element
            signal = Signal
            human_onsets = np.loadtxt(Human_onset, dtype='float')
        else:
            human_onsets = np.loadtxt(human_onset, dtype='float')
        if fps==100:
            for j in range(0, limit): 
                for i in range(0,100):
                    peaks_detector = OnsetPeakPickingProcessor(fps=fps, threshold = j+ i/100) #create a peak detector 
                    peaks = peaks_detector(onset_processor(signal))

                    evaluation = OnsetEvaluation(peaks, human_onsets, window=window_size)
                    if evaluation.fmeasure >= best_f1score:
                        threshold = j+ i/100
                        best_f1score = evaluation.fmeasure
        else:
            framed_signal = FramedSignal(signal, fps=fps)
            stft_signal = ShortTimeFourierTransform(framed_signal)
            log_filt_spec = LogarithmicFilteredSpectrogram(stft_signal, num_bands=24, fps=fps)
            for j in range(0, limit): 
                for i in range(0,100):
                    peaks_detector = OnsetPeakPickingProcessor(fps=fps, threshold = j+ i/100) #create a peak detector 
                    peaks = peaks_detector(onset_processor(log_filt_spec))
                    evaluation = OnsetEvaluation(peaks, human_onsets, window=window_size)
                    if evaluation.fmeasure >= best_f1score:
                        threshold = j+ i/100
                        best_f1score = evaluation.fmeasure
        threshold_list.append(threshold)
        if(not isinstance(Signal, list)): # if List only has one element
            break
    print(best_f1score)
    return threshold_list

def onset_detection(audio_file, method, lim):
    """
        Objective: find onset of the given audio file
             args: audio_file: audio file name
           method: Onset detecton method function
              lim: threshold of peak picking
            
           return: temporal position of each onset
    """
    sig = Signal(audio_file) # Transform audio to signal
    peaks_detector = OnsetPeakPickingProcessor(fps=100, threshold =lim) # Create a onset peak detector 
    peaks = peaks_detector(method(sig)) # Find peaks
    return peaks

def pitch(x, fs):
    """
        Objective: Compute the pitch of a signal x using autocorrelation method.
             args: x: audio signal
                  fs: audio signal sample frequency  
           return: f0: detected frequency of the signal
        
    """
    N=len(x)
    acorr = librosa.autocorrelate(x)
    indexMax = np.argmax(acorr)
    xf,yf=do_fft(x, fs)
    k=20
    const = 1/k
    while(True):
        peaks, _ = find_peaks(acorr, height=acorr[indexMax]-const*acorr[indexMax])
        k-=1
        if k==0:
            const = 1
        else:
            const=1/k
        if len(peaks) >2 or const == 1:
            break
    if len(peaks)==1 or len(peaks)==0:
        lag = np.argmax(acorr[3:])+3
    else:
        lag = peaks[1]-peaks[0]
    
    f0 = fs/lag
    #print("freq", f0)
    return f0

def get_rms(x, frame_size=0, hop_size=0):
    """Compute the RMS of a signal in frames.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    frame_size : int
        Frame size in samples. If 0, the whole signal is used.
    hop_size : int
        Hop size in samples. If 0, the frame size is divided by 4.
    
    Returns
    -------
    rms : np.ndarray
        RMS of the signal in frames.
    """

    if frame_size == 0:
        frame_size = len(x)
        hop_size = frame_size
    if hop_size == 0:
        hop_size = frame_size/4
    
    start = 0
    end = frame_size
    fbrms=[]
    while(True):
        frame = x[start:end]
        sum = 0
        for frame_n in frame:
            sum += frame_n**2
    
        frame_rms = np.sqrt(sum/frame_size)
        fbrms.append(frame_rms)
        start += hop_size
        if end+hop_size<= len(x):
            end += hop_size
        else:
            break

    return fbrms

def get_duration(signal, fs, onsets):
    """
        Objective: define the duration of each note as diference between onsets
             args: signal: input audio signal
                       fs: sampling frequency
                   onsets: onsets of the signal
            return: note durations
    """
    duration_audio = len(signal)*(1/fs)
    note_duration = []
    for i in range(0,len(onsets)):
        if i == len(onsets)-1:
            # get each note segment 
            note_duration.append(duration_audio-onsets[i])
        else:
            note_duration.append(onsets[i+1]-onsets[i])
    return note_duration

def find_digit(string):
    for d in string:
        if d.isdigit():
            return int(d)
        
def get_notes(signal, fs, onsets):
    """
        Objective: find musical note in a given signal
             args: signal: input audio signal
                       fs: sampling frequency
                   onsets: onsets of the signal
            return: detected notes
    """
    duration_audio = len(signal)*fs
    note_segment = []
    note = []

    for i in range(0,len(onsets)):
        window_start=0
        window_end=0.1
        #print(i)
        while(True):
            if i==len(onsets)-1:
                note_segment = signal[int((onsets[i]+window_start)*fs):int(duration_audio*fs)]
            else:
                note_segment = signal[int((onsets[i]+window_start)*fs):int((onsets[i]+window_end)*fs)]
            #print(len(note_segment))
            note_freq = pitch(note_segment, fs)
            n = librosa.hz_to_note(note_freq)
            if i==0:
                last_note=n
            # find digit to compare octave
    #        print("last",last_note,"   now",n)
            
            l_octave = find_digit(last_note)
            n_octave = find_digit(n)
            window_start+=0.001
            window_end+=0.001
            # two octave of difference is too much, probably the found note is incorrect
            if abs(l_octave-n_octave)<=1 or onsets[i]+window_end > duration_audio:
                #print(l_octave, n_octave)
                break 
        #if find the correct note append

        note.append(n)
        last_note=n
    return note


def make_composition(notes, amplitude, duration, onsets):
    """
        Objective: Make a composition vector as [note, amplitude, duration, onsets]
             args: audio_file: audio file name

            return: musical composition
    """
    composition=[]
    for n,a,d,o in zip(notes, amplitude, duration, onsets):
        composition.append([n,a,d,o])
    return composition

def get_amplitude(signal, fs, onsets, frame_size=0, hop_size=0, plot=False):
    """
    Objective: get the amplitude value of each note based on Frame-based root mean square (RMS)
            args: signal: input audio signal
                    fs: audio signal sampling frequency
            frame_size: Frame size in samples. If 0, the whole signal is used.
                hop_size: Hop size in samples. If 0, the frame size is divided by 4.
                    plot: if True plot the variation of rms 
            
                return: all notes amplitudes 
    """
    duration_audio =  len(signal)*fs
    
    note_segment = []
    note_amplitude =[]

    total_rms = []
    for i in range(0, len(onsets)):
        rms_list = []
        if i == len(onsets)-1:
            # get each note segment 
            note_segment = signal[int(onsets[i]*fs):int(duration_audio*fs)]
        else:
            # get each note segment 
            note_segment = signal[int(onsets[i]*fs):int(onsets[i+1]*fs)]
        # get rms variations
        rms_list = get_rms(note_segment, frame_size=frame_size, hop_size=hop_size)
        # find rms maximum value
        max_rms = max(rms_list)
        note_amplitude.append(max_rms)
        # insert on total rms list (usefull in plot)
        for rms in rms_list:
            total_rms.append(rms)

    if(plot):
        t = np.arange(len(total_rms)) * hop_size / fs
        plt.plot(t, total_rms)
        librosa.display.waveshow(signal, sr=fs)

    return note_amplitude

def do_fft(x, fs):
    """
        Objective: find frequency domain of x
             args: x: input signal(time domain)
                tini: time start of conversion 
                tend: time end of the conversion
          return: xf: frequency of x in given interval
                yfdB: amplitude of each frequency in decibeis
    """
    # sample spacing( sampling period)
    T = 1.0 / fs 
    N_total = len(x)
    # amplitude of fft
    yf = rfft(x)
    yfdB = librosa.amplitude_to_db(np.abs(yf), ref=np.max)
    #frequeny of fft
    xf = rfftfreq(N_total, T)
    return xf, yf

"""
LAB1
Evaluation of Onset Methods

"""

"""
    Analyse the performance of the algorithm 

    args: audios -> vector of audios file name
          HumanOnsest -> vector of Human annotation onset file name
          method -> algorithem to be analysed
          SeeGraph -> boolean value, plot or not the graphs

    return: average f-measurement of all audios
    
"""      
def onset_analyse_performance(audios, HumanOnsets, thresholds, method, SeeGraph):
    evaluation = []
    for audio, humanonset,thresholds in zip(audios,HumanOnsets, thresholds):
        evaluation.append( method(audio, humanonset, thresholds, SeeGraph))
    
    performance = []
    avg_fmeasure = 0
    for audio,evalu in zip(audios,evaluation):
        performance.append([audio.rsplit("/",1)[-1], evalu.num_tp,
                           evalu.num_fp, evalu.num_fn, 
                           evalu.precision, evalu.recall,
                           evalu.fmeasure])
        avg_fmeasure += evalu.fmeasure
    N_audios = len(audios)
    
    avg_fmeasure = avg_fmeasure/N_audios
    
    print(tabulate(performance, headers=["Recording","TP", "FP", "FN", "Precision", "Recall", "F-measure"]))
    print("\nAvarage F-measure : "+str(round(avg_fmeasure,3)))
    
    return avg_fmeasure
"""
    Compare The Human Anotation onset and Peaks detected by the algorithm
    
    args: peaks -> Automatic onset annotation 
          human_onset -> Human annotation onset file name
       
    Return: class with values os f_measurement, precision, recall, etc...
"""
def get_evaluation(human_onset, peaks):
    # EVALUATION
    human_onsets = np.loadtxt(human_onset, dtype='float')
    evaluation = OnsetEvaluation(peaks, human_onsets, window=0.100)
    return evaluation


"""
    CNN method
    
    args: audios -> vector of audios file name
          HumanOnsest -> vector of Human annotation onset file name
          lim -> threshold
          do_plot -> boolean value, plot or not the graphs
    
    return: class with values os f_measurement, precision, recall, etc...
    
""" 

def onset_CNN(audio, human_onset, lim, do_plot):
    x = Signal(audio) 
    CNN_act = CNNOnsetProcessor() #set CNN onset processor
    
    peaks_detector = OnsetPeakPickingProcessor(fps=100, threshold =lim) #Create a onset peak detector 
    peaks = peaks_detector(CNN_act(x))    # find peaks

    #plots
    if(do_plot):
        plot_audio_onsets(audio, peaks, human_onset)
    
    return get_evaluation(human_onset, peaks)

"""
    RNN method
    
    args: audios -> vector of audios file name
          HumanOnsest -> vector of Human annotation onset file name
          lim -> threshold
          do_plot -> boolean value, plot or not the graphs
          
    return: class with values os f_measurement, precision, recall, etc...
""" 


def onset_RNN(audio, human_onset, lim, do_plot):
    x = Signal(audio) #transform to audio to Signal
    RNN_act = RNNOnsetProcessor() # create a RNN processor
    peaks_detector = OnsetPeakPickingProcessor(fps=100, threshold =lim) #Create a onset peak detector 
    peaks = peaks_detector(RNN_act(x))    # find peaks
    
    #plots
    if(do_plot):
        plot_audio_onsets(audio, peaks, human_onset)
    
        
    return get_evaluation(human_onset, peaks)

"""
    SuperFlux method
    
    args: audios -> vector of audios file name
          HumanOnsest -> vector of Human annotation onset file name
          lim -> threshold
          do_plot -> boolean value, plot or not the graphs
    
    return: class with values os f_measurement, precision, recall, etc...
    
""" 
def onset_superFlux(audio, human_onset, lim, do_plot):
    #IMPORT DATA
    x = Signal(audio) 
    # ONSET DETECTION
    framed_x = FramedSignal(x, fps=200)
    stft_x = ShortTimeFourierTransform(framed_x)
    log_filt_spec = LogarithmicFilteredSpectrogram(stft_x, fps=200, num_bands = 24)
    SuperFlux_acti = superflux(log_filt_spec, diff_max_bins=3) # superflux algotithm

    peaks_detector = OnsetPeakPickingProcessor(fps=200, threshold = lim) #set a peak detector 
    peaks = peaks_detector(SuperFlux_acti)

    if(do_plot):
        plot_audio_onsets(audio, peaks, human_onset)
        
    return get_evaluation(human_onset, peaks)

"""
    Complex Flux method
    
    args: audios -> vector of audios file name
          HumanOnsest -> vector of Human annotation onset file name
          lim -> threshold
          do_plot -> boolean value, plot or not the graphs
          
    return: class with values os f_measurement, precision, recall, etc...
    
"""

def onset_complexFlux(audio, human_onset, lim, do_plot):
    #IMPORT DATA
    x = Signal(audio) 

    # ONSET DETECTION
    framed_x = FramedSignal(x, fps=200)
    stft_x = ShortTimeFourierTransform(framed_x)
    log_filt_spec = LogarithmicFilteredSpectrogram(stft_x, num_bands=24, fps=200)
    acti = complex_flux(log_filt_spec) # superflux algotithm

    peaks_detector = OnsetPeakPickingProcessor(fps=200, threshold = lim) #set a peak detector 
    peaks = peaks_detector(acti)

    if(do_plot):
        plot_audio_onsets(audio, peaks, human_onset)
                 
    return get_evaluation(human_onset, peaks)

    
"""
    Plot the Spectrogram, and visualize the differences between
    Human Anotation onset and Peaks detected by the algorithm
    
    args: audio -> audio file name
          human_onset -> Human annotation onset file name
          Machine_onset -> Automatic onset annotation 
"""     
def plot_audio_onsets(audio, Machine_onset, Human_onset):
    human_onsets = np.loadtxt(Human_onset, dtype='float')  #load human writed onsets
    
    x, fs = librosa.load(audio) # get audio
    D = np.abs(librosa.stft(x))
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', ax=ax[0])
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()
    
    librosa.display.waveshow(x, sr=fs, ax=ax[1])
    librosa.display.waveshow(x, sr=fs, ax=ax[2])
    plt.title("Linear graph")
    
    ax[1].vlines(Machine_onset, -1, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[1].set_title("Automatic Onset Detection" + audio.rsplit("/",1)[-1])
    ax[2].vlines(human_onsets, -1, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[2].set_title("Human Onset Annotation" + audio.rsplit("/",1)[-1])