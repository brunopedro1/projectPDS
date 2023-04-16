import numpy as np
from IPython.display import Audio
from matplotlib import pyplot as plt
import librosa
import librosa.display
import math
from scipy.signal import square 
from scipy.signal import sawtooth 

def note_octave_change(note, value):
    """
        change octave of a note depending on value
            new octave = old octave + value
    """
    note = list(note)
    for i in range(0, len(note)):
        if note[i].isdigit():

            note[i] = str(int(note[i])+value)
            break;
    
    return "".join(note);
def plot_audio(x, spectrogram=True):
    """
        plot_audio():
            Description: plot audio spectogram and wave form
            args: x: sampled audio
            spectogram: boolean, visualize or not the spectogram
    """
    fs =22050
    D = np.abs(librosa.stft(x))

    if spectrogram:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1]})
        img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),sr=fs, x_axis='time',y_axis='linear',ax=ax[0])
        ax[0].set(title='Power spectrogram')
        ax[0].label_outer()
        librosa.display.waveshow(x, sr=fs, ax=ax[1])
    else:
        fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(9, 2.5))
        librosa.display.waveshow(x, sr=fs, ax=ax)
    plt.tight_layout()
    plt.show()

def note_frequency(note):
    """
        Find note frequency
    """
    semitones = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    semitones_list = list(range(0,len(semitones)))
    note_splitted = list(note)
    
    fA4 = 440

    # Find note number index
    for s in note:
        if s.isdigit():
            note_number = int(s)
        else:
            continue

    # Find note letter
    for i in semitones_list:
        if note_splitted[0] == semitones[i]:
            note_letter = i
        else:
            continue
    
    # Calculate note frequency
    frequency = (fA4*2**((note_letter-semitones.index('A'))/12+(note_number-4)))
    if '#' in note:
        frequency *= 2**(1/12)
    return frequency

class synthesizer:
    """
        Audio synthesizer
            Realize Polyphonic Synthesis. 
            The input composition should be a vector with ["note", "Amplitude", "duration", "start time"].   
    """
    def __init__(self,fs):
        self.dur=64
        self.wtable = np.sin    
        self.attack = 0.1
        self.decay=0.2
        self.sustain=0.8
        self.release=0.4
        self.height=1
        self.fs =fs
    

    def define_adsr_envelope(self,attack, decay, sustain, release, height=1):
        """
        Func: define_adsr_envelope()
            Objective: sets parameters of envelope
        """
        self.attack=attack
        self.decay = decay
        self.sustain=sustain
        self.release=release
        self.height=height
    

    def wavetable_generate(self, funct, dc=0.5, width=1):
        """
        Func:  wavetable_generate()
            Objective: Generate a wavetable of dur samples using the specified
                      function that is assumed to have a period of 2pi
            Args: funct: wave form
                  dc: duty cicle of squared wave
                  width: Width of the rising ramp as a proportion of the total cycle in the sawtooth wave
        """ 
        result = np.array([])
        if(funct == square):
            for i in range(0, self.dur):
                result = np.append(result, funct(2*np.pi*i/self.dur, dc))
                
        elif(funct == sawtooth):
            for i in range(0, self.dur):
                result = np.append(result, funct(2*np.pi*i/self.dur, width))
                
        else:
            for i in range(0, self.dur):
                result = np.append(result, funct(2*np.pi*i/self.dur))
        self.wtable = result

    def define_beat(self, composition, new_beat, old_beat):
        """
        Func:  define_beat()
            Objective: Modify the beat of audio file
                Args: audio: audio_file class 
                        beat: new beat value
                Return: new audio_file class 
        """ 
        ratio = old_beat/new_beat
        lenght = len(composition)
        beat_comp = np.array(composition)
        for i in range(0,lenght):
            duration = float(composition[i][2])
            t_ini = float(composition[i][3])
            duration *= ratio
            t_ini *= ratio
            beat_comp[i][2] = duration
            beat_comp[i][3] = t_ini

        return beat_comp


    def set_wavetable(self, wt):
        """
        Func:  set_wavetable()
            Objective: set new wavetable in synthetizer
                Args: wt: wavetable
        """ 
        self.wtable = wt


    def add_harmonic(self, synth_audio, harmonics):
        """
        Func: add_harmonic()
            Objective: add a harmonic serie to the audio vector
                Args: synth_audio: audio vector
                      harmonics: vector with fundamental note and amplitude
              return: cancatunation of audio file and harmonic serie
        """ 
        fundamental_freq = note_frequency(harmonics[0]) #get note freq
        amplitude = harmonics[1]
        self.wavetable_generate(np.sin) #generate sin wavetable
        for i in range(1, 100): #create harmonic serie with N=100
            freq=fundamental_freq*i
            swav = amplitude*self.wavetable_synthesis(self.wtable, freq, self.fs, synth_audio.size/self.fs) 
            amplitude/=1.2
            synth_audio=np.add(swav, synth_audio)

        return synth_audio


    def synthesize(self, composition):
        """
        Func: composition()
            Objective: Synthesize a sound signal based on a symbolic representation of a
                      musical composition
                Args: audio: audio vector
              return: synthetized audio
        """ 
        # Find the total size of the composition (in seconds and number of samples)
        duration = 0
        i = 0
        size_seconds = 0
        for note, amplitude, duration, t_ini in composition:
            duration=float(duration)
            t_ini=float(t_ini)
            if t_ini+duration >= size_seconds:
                size_seconds = t_ini+duration
                size_samples = int(size_seconds*self.fs) 

        # Compute synthesized composition
        out = np.zeros(size_samples)
        for note, amplitude, duration, t_ini in composition:
            duration=float(duration)
            t_ini=float(t_ini)
            amplitude = float(amplitude)
            freq = note_frequency(note)
            swav = amplitude*self.wavetable_synthesis(self.wtable, freq, self.fs, duration)
            adsr_wav = self.adsr_envelope(swav)

            sample_ini = int(t_ini*self.fs)
            out[sample_ini:len(adsr_wav)+sample_ini] += adsr_wav
            
        return out 


    def wavetable_synthesis(self, wavetable, freq, sr, dur):
        """
        Synthesize a periodic signal using a given wavetable.
        The wavetable is assumed to have a period of 2pi.
        """
        # N is the number of samples of the synthesized signal
        N = int(dur*sr)
        # L is the length of the wavetable
        L = wavetable.size
        # N0 is the desired period in samples of the synthesized signal
        N0 = int(sr/freq+0.5)
        # buffer signal of period L with the right number of periods
        buffer = np.tile(wavetable,int(np.ceil(N/N0)))
        # resample buffer to match the desired period with linear interpolation
        # using resampling coefficient a=L/N0
        s = np.interp(np.arange(0, buffer.size, L/N0), np.arange(buffer.size), buffer)
        return s[:N]
    

    def adsr_envelope(self, x):
        """
        Shape a signal with an ADSR envelope.
        attack: ratio of the attack phase
        decay: ratio of the decay phase
        sustain: amplitude ratio of the sustain phase
        release: ratio of the release phase
        height: amplitude ratio of the whole signal
        Attack + decay + release can't be more than 1 or error will occour
        if equal to 1 no sustain will happen
        """
        if( (self.attack+self.decay+self.release) > 1):
            print("Error: Attack + decay + release can't be more than 1\n")
            return 
        shapeA = np.linspace(0, 1, int(self.attack * len(x)))
        shapeD = np.linspace(1, self.sustain, int(self.decay * len(x)))
        shapeR = np.linspace(self.sustain, 0, int(self.release * len(x)))
        shapeS = np.ones(len(x)-len(shapeA)-len(shapeD)-len(shapeR)) * self.sustain
        shape = np.concatenate((shapeA, shapeD, shapeS, shapeR))
        return x * shape * self.height
    

    def delay_t(self, composition, t_delay):
        delay_composition = list(composition).copy()
        for i in range(0, len(delay_composition)): # sum the start of note delay value
            delay_composition[i]= list(delay_composition[i])
            delay_composition[i][3]= delay_composition[i][3]+t_delay
            
        delay_composition.insert(0,["C4",0,t_delay,0]) # insert delay on the start
        
        return delay_composition
    
    def octave_change(self, composition, change=1):
        """
         UP or DOWN the octave of a composition depending on change value
            if change >0, increase the octave by its value
            if change <0, decrease the octave bt its value
        """
        new_comp = list(composition).copy()
        for i in range(0, len(new_comp)):
            new_comp[i] = list(new_comp[i])
            new_comp[i][0] = note_octave_change(new_comp[i][0], change)

        return new_comp
    
    def add_two_signal(self, signal1, signal2):
        if len(signal1) > len(signal2):
            sig1 = signal1.copy()
            sig2 = signal2.copy()
        else:
            sig1 = signal2.copy()
            sig2 = signal1.copy()
        
        lenght_sig2 = len(sig2)
        lenght_sig1 = len(sig1)
        
        for i in range(lenght_sig2, lenght_sig1):
            sig2=np.append(sig2, 0)
        
        finalsignal = np.add(sig1, sig2)

        return finalsignal

    def piano(self, composition, fs):

        synth = np.zeros(int(fs*(composition[-1][3]+composition[-1][2]+0.5)), dtype=float) # 0.5s of tolerance

        for note, amplitude, duration, t_ini in composition:
            duration=float(duration)
            t_ini=float(t_ini)
            amplitude = float(amplitude)
            
            if(len(note) ==3 ):
                note = list(note)
                note[1] = '#'

            note_audio, _ =librosa.load('projectPDS/recordings/notes/{}.wav'.format("".join(note)),sr=fs)
            for i in range(0, int((duration+0.05)*fs)):
                synth[int(t_ini*fs)+i]+=amplitude*note_audio[i]

        return synth
    
