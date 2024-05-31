#%%
# Import needed packages
import pyvisa
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from scipy.io.wavfile import read, write
from matplotlib.font_manager import FontProperties
from colorama import Fore, Back, Style
import pandas as pd
from RSRTxReadBin.RTxReadBin import RTxReadBin # File has been modified to accept the number of channels as a parameter when loading .bin data files. 
                                               # To do this, find the find in your local Python environment after installing the .whl file
                                               # Then add the "ch" argument to the reading function. 
import os

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft

# Remove syntax warnings
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)


def Get_resources():
    '''Get resources from pyvisa and list them'''
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    return rm

# ═══════════════════════════════════
# ════ Osciloscope ══════════════════
# ═══════════════════════════════════

class oscilloscope():
    def __init__(self, timeout = 25000):
        ''' Open connection to oscilloscope and do basic config'''
        print("Initialization of oscilloscope")
        # Open connection to oscilloscope
        self.address = 'USB0::0x0AAD::0x0197::1320.5007k16-103151::INSTR'
        
        self.rm = pyvisa.ResourceManager()
        self.osc = self.rm.open_resource(self.address)
        try:
            if self.osc.query("*IDN?") != "":
                print("Oscilloscope ID: ", self.osc.query("*IDN?"))
            else: 
                print("Oscilloscope not connected, No ID")
                return
        except:
            print("Oscilloscope not connected, shit")
            return
        
        # Initial settings
        self.osc.write("*RST")                   # Reset oscilloscope
        time.sleep(1)                       # Wait for oscilloscope to reset
        #self.osc.write("WAI")
        self.osc.write("FORMat:DATA REAL,32")    # Set data format to 32 bit real
        # purge timeout
        self.osc.timeout = timeout

        # run setup function
        self.setup()
    
    
    
    # ~~~○○○ Functions for setting up oscilloscope ○○○~~~
    def setup(self, 
    hscale = 4e-9, 
    vscale = 5e-3, 
    res = 100e-12,
    trigger_level = 0.5,
    trigger_source = 2,
    offset_time = 4
    ):
        '''set hoizontal and veritcal scale etc'''
        print(f"Setting horizontal scale {hscale} and vertical scale {vscale} etc")
        self.osc.write("SYSTem:DISPlay:UPDate ON") # Turn on display
        
        # for channel 1
        
        self.osc.write("CHANnel1:DISPlay ON")  # Turn on channel 1
        self.osc.write(f"TIMebase:SCALe {hscale}")
        self.osc.write(f"CHANnel1:SCALe {vscale}")
        

        # for channel 2
        self.osc.write("CHANnel2:DISPlay ON")  # Turn on channel 2
        self.osc.write("CHANNEL2:WAVEFORM:STATE ON")
        self.osc.write(f"CHANnel2:SCALe 0.2")
        

        
        # trigger setup
        self.osc.write(f"TRIG:SOUR {trigger_source}")
        #self.osc.write(f"CHANnel{trigger_source}:SCALe 0.5")  # Set scale for trigger channel
        self.osc.write(f"ACQuire:RESolution {res}")
        self.osc.write(f'TRIG:LEV{trigger_source} {trigger_level}')
        self.osc.write("TRIG:MODE NORMal")
        oft = offset_time * hscale
        self.osc.write(f"TIMebase:HORizontal:POSition {oft}")  # Adjusts offset time

        self.osc.write(f"TRIG:LEV5 {trigger_level}")
        
        
        
    # # ~~~○○○ Functions for managing oscilloscope states ○○○~~~
    
    # def save_state(self, save_number=10):
    #     '''Save oscilloscope state'''
    #     if self.osc.query("*OPC?") == "0":               # Check if oscilloscope is busy
    #         warnings.filterwarnings('ignore', category=SyntaxWarning)
    #         self.osc.write(f"*SAV {save_number} 'C:\Settings\state{save_number}.dfl'")
    #         self.osc.write(f"MMEMory:STORe:STATe {4}")   # Save state
    #     else: 
    #         print("Oscilloscope busy...")
    
    # def load_state(self, save_number=10):
    #     '''Load oscilloscope state'''
    #     if self.osc.query("*OPC?") == "0":               # Check if oscilloscope is busy
    #         warnings.filterwarnings('ignore', category=SyntaxWarning)
    #         self.osc.write(f"MMEMory:LOAD:STATe {4} 'C:\Settings\state{save_number}.dfl'")     
    #         self.osc.write(f"*RCL {save_number}")        # Activate loaded state
    #     else: 
    #         print("Oscilloscope busy...") 


    # ~~~○○○ Functions for taking measuments ○○○~~~
    
    def meas_amp(self, ch=1, avg=1):
        '''Measure amplitude of the signal'''
        
        if self.osc.query("*OPC?") == "0":               # Check if oscilloscope is busy
            self.osc.write(f"MEAS1:SOUR C{ch}W1")        # Set source to channel 1
            self.osc.write("MEAS1:MAIN AMPL")            # Set measurement to amplitude
            self.osc.write("MEAS1 ON")                   # Turn on measurement
            self.osc.query("*OPC?")                      # Wait for measurement to complete
            return self.osc.query("MEAS1:RES:ACT?")      # Return the result
        else: 
            print("Oscilloscope busy...")
        
    
    # ~~~○○○ Run single amplitude measurement ○○○~~~
    def single_meas(self, ch=1, avg=1, savewhere = "pc", localsavepath = r""):
        '''
        Run single amplitude measurement
        returns numpy array with waveform data points
        '''
        #print(f"Amplitude of channel {ch} is {self.osc.meas_amp(ch, avg)} V")
        #if savewhere != ("pc" or "local"):
        #    print("YOU IDIOT")
        #    return
        if self.osc.query("*OPC?") == "1\n":               # Check if oscilloscope is busy
            self.osc.write(f"MEAS1:SOUR C{ch}W1")   # set channel
            self.osc.write("MEAS1:MAIN AMPL")       # set measure amplitude
            self.osc.write(f"ACQuire:COUNt {avg}")
            #if avg != 1:                            # Set # of averages (Minimum 10)
            #self.osc.write(f"ACQuire:AVERage:COUNT {avg}")  
            
            self.osc.write("SINGle")                  # Run single measurement
            #if avg != 1:
            self.osc.write("*WAI")

            # Get the data:
            # Choose format
            self.osc.write("FORM ASC")          # formating
            # Remove axis data
            self.osc.write("EXP:WAV:INCX OFF")  # no time scale
            self.osc.write("*WAI")
            if savewhere == "pc":
                raw_data = self.osc.query(f"CHAN{ch}:WAV1:DATA?")      # Return waveform
                np_listed_float_data = np.array(raw_data.split(",")).astype(float)
                return np_listed_float_data
            
            if savewhere == "local":
                self.osc.write("EXPORT:WAVEFORM:FASTEXPORT ON")
                self.osc.write("EXPort:WAVeform:MULTichannel ON")
                self.osc.write(f"EXPORT:WAVEFORM:NAME '{localsavepath}.bin'") # set path for save

                self.osc.write("EXPORT:WAVEFORM:SAVE") # DO

                return
        else:
            print("Oscilloscope busy...")
            

# ═══════════════════════════════════
# ════ Waveform generator ═══════════
# ═══════════════════════════════════

class waveGen():
    def __init__(self):
        ''' Open connection to waveform generator and do basic config'''
        print("Init of waveform generator")
        # Open connection to waveform generator slow
        # self.address = 'USB0::0x0957::0x2807::MY59001304::INSTR'
        
        # Open connection to waveform generator fast
        self.address = 'USB0::0x0957::0x2C07::MY52813688::0::INSTR'
        
        self.rm = pyvisa.ResourceManager()
        
        try:
            self.waveGen = self.rm.open_resource(self.address)
            if self.waveGen.query("*IDN?") != "":
                print("Waveform generator ID: ", self.waveGen.query("*IDN?"))
            else: 
                print("Waveform generator no ID returned")
        except:
            print("Waveform generator not connected")
        
        # Initial settings
        self.waveGen.write("*CLS")               # Clear status
        self.waveGen.write("*RST")               # Reset waveform generator
        time.sleep(1)                       # Wait for wavegen to reset
        print("Waveform generator initialized with adress:", self.address)
        self.waveGen.write("OUTP:SYNC OFF")
    
    
    # ~~~○○○ Functions for resetting waveform generator ○○○~~~
    def reset(self):
        '''Reset waveform generator'''
        self.waveGen.write("*RST")               # Reset waveform generator  
        self.waveGen.write("OUTP:SYNC OFF")
        time.sleep(0.2)                       # Wait for wavegen to reset 
        print("Waveform generator reset")    
    
    
    # ~~~○○○ Functions for applying sinewave ○○○~~~
    def sin(self, freq=1000, amp=1, offset=0, load=50, t=1000):
        '''Apply sinewave to output      
            \nfreq - frequency in Hz
            \namp - amplitude in Vpp
            \noffset - offset in V
            \nload - load in ohm
            \ntime - time in ms    '''
        self.waveGen.write("SOURCE1:FUNCTION SIN")           # Set sine wave
        self.waveGen.write(f"SOURCE1:FREQUENCY {freq}")      # Set frequency
        self.waveGen.write("SOURCE1:VOLT:UNIT VPP")          # Set voltage unit to Vpp
        self.waveGen.write(f"SOURCE1:VOLT {amp}")            # Set amplitude
        self.waveGen.write(f"SOURCE1:VOLT:OFFSET {offset}")  # Set offset
        self.waveGen.write(f"OUTPUT1:LOAD {load}")           # Set load
        
        self.waveGen.write("OUTP:SYNC ON")                  #
        self.waveGen.write("OUTPUT1 ON")                     # Turn on output
        print(f"Applying sine with frequency {freq} Hz, amplitude {amp} Vpp, offset {offset} V and load {load} ohm")
        if t == 0:
            print("Indefinite time")
            return
        else:
            print(f"Running for {t}ms")
            time.sleep(t/1000)                       # Wait for time
            self.waveGen.write("OUTPUT1 OFF")                # Turn off output
            self.waveGen.write("OUTP:SYNC OFF")
       
    # ~~~○○○ Functions loading arbitrary waveforms ○○○~~~
    def arb(self, path="", voltage=1, srate=1e6 , load=50, penis = True):
        '''Load arbitrary waveform from file to output
        \npath - path to file, ex. /arb/test.arb
        \nvoltage - amplitude in Vpp
        \nsrate - sample rate in Hz
        \nload - load in ohm'''
        
        if path == "":
            print("No path given")
            return
                
        else:
            
            self.waveGen.write("SOURCE1:FUNCTION ARB")                    # Set arbitrary wave
            self.waveGen.write(f'MMEM:LOAD:DATA "USB:{path}"')
            time.sleep(0.5)
            self.waveGen.write(f"SOURCE1:FUNCtion:ARB:SRATe {srate}")     # Set sample rate
            self.waveGen.write(f"OUTPUT1:LOAD {load}")                    # Set output load
            time.sleep(0.1)
            self.waveGen.write(f'SOUR1:FUNC:ARB "USB:{path}"')
            self.waveGen.write(f"SOURCE1:VOLTAGE {voltage} Vpp")
            #self.waveGen.write(f"SOURCE1:FUNC:ARB:PTP {voltage}")         # Set amplitude
            time.sleep(0.1)
            self.waveGen.write("OUTP ON")
            self.waveGen.write("OUTP:SYNC ON")
            
            err = self.waveGen.query("SYST:ERR?")
            
            if penis:
                print(f"Arbitrary waveform loaded from {path}")
                if err != ' +0,"No error"':
                    print(Fore.RED + "WAVEGEN ERRORS: ", err, Style.RESET_ALL)
                else:
                    print(Fore.GREEN + "No errors", Style.RESET_ALL)
                
            
            
                
                
    
    # ~~~○○○ Functions for turning off output ○○○~~~
    def off(self):
        '''Turn off output'''
        self.waveGen.write("OUTPUT1 OFF")                # Turn off output
        self.waveGen.write("OUTP:SYNC OFF")
        self.waveGen.write("DATA:VOL:CLEAR")
        print("Output turned off")


# ═══════════════════════════════════
# ════ Data management functions ════
# ═══════════════════════════════════

def generateArb(file, save_path, samples=5000):
    ''''Generate arbitrary waveform from wav-file. 
        \nPath - Complete path to wav-file
        \nSave_path - Place to save file, name autogenerated
        \nSamples - number of samples in total'''

    # Load the data from the wav-file and normalize to +-32767
    spoken_digit_read = read(file)[1] # get array not samplingrate
    spoken_digit = np.copy(spoken_digit_read).astype("float")
    spoken_digit *= 1/(np.max(np.abs(spoken_digit)))
    spoken_digit = (32767*spoken_digit).astype(int)

    # Check if the file is too short
    if len(spoken_digit) > samples:
        print(f"Too few samples selected, needs at least {len(spoken_digit)} samples")
        return

    with open(f"{save_path}/{file.split('/')[-1].split('.')[0]}.arb","w") as f:
        # Create the file header needed
        f.write("File Format:1.10"+ "\n")
        f.write("Channel Count:1"+ "\n")
        f.write(f"Data Points:{samples}"+ "\n")
        f.write("Data:"+ "\n")
        
        # Add the data from the loaded wav-file
        for sample in spoken_digit:
            f.write(str(sample) + "\n")
        
        # Add zeros to fill up the file
        z = samples - len(spoken_digit)  
        for i in range(z):
            f.write("0" + "\n")
            
    print(f"Arb generated for:{file.split('/')[-1].split('.')[0]}", 
          f"\nat: {save_path}\n" )
    
    
    
## get fmax helper function
def get_f_max(t, my, cutoff = 200e6, zeropad = False, no_dc = True):

    zero_pad_N = 100000

    dt = t[1]-t[0]
    
    N = len(t)
    if zeropad:
        # FFT code
        extra = np.zeros((zero_pad_N))
        mye = np.concatenate((my,extra))
        freq = rfft(mye-np.mean(mye))
        f = rfftfreq(N+zero_pad_N, dt)
    else:
        # FFT code
        freq = rfft(my-np.mean(my))
        f = rfftfreq(N, dt)
    #plt.plot(f,freq)
    
    idx = [i for i,v in enumerate(f) if v > cutoff]

    freq = freq[idx]
    f = f[idx]

    f_max = f[np.argmax(np.abs(freq))]

    #if f_max < 2e9:
    #    plt.plot(f[:50],freq[:50])
    return f_max

# ═══════════════════════════════════
# ════ Speech classifier═════════════
# ═══════════════════════════════════


class speechClassifierLab():
    def __init__(self, data_folder, feature_space = 20, cutoff = 200e6, processing = True):
        
        files = []
        for f in os.listdir(data_folder):
            if f.endswith("Wfm.bin") and (f.startswith("1") or f.startswith("0") or f.startswith("2")
                                          or f.startswith("3") or f.startswith("4") or f.startswith("5")
                                          or f.startswith("6") or f.startswith("7") or f.startswith("8")
                                          or f.startswith("9")):
                files.append(f)
        self.feature_space = feature_space
        self.features = np.zeros((len(files), feature_space))
        self.classes = []
        
      
        for n, file in tqdm(enumerate(files), total=len(files)):
            wfm_data, b, meta_data = RTxReadBin("../Lab/Classification/FullRunBin/"+file, nNofChannels=2)
            wfm_data  = np.array(wfm_data[:,0,0]) # sindsyg formatering
            t = np.linspace(0, len(wfm_data)*meta_data["Resolution"], len(wfm_data))
            #append class from filename
            self.classes.append(int(file[0]))
            #print(f"Processing file {n+1} of {len(files)}")
        
            for i,(ti,wfmi) in enumerate(zip(np.array_split(t,feature_space), np.array_split(wfm_data,feature_space))):
                self.features[n,i] = get_f_max(ti, wfmi, cutoff=cutoff)
        
    def clf(self, layers = (10,10)):
        X = self.features
        # XOR target labels
        y = self.classes
        # Create an MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=layers, max_iter=100000, random_state=42,activation="tanh")
        
        # Train the model
        clf.fit(X, y)

        # Make predictions
        predictions = clf.predict(X)
        #print("Predictions:", predictions)

        # Print accuracy
        accuracy = np.mean(predictions == y)
        print("Accuracy:", accuracy)
        return accuracy
    
    def PCA_plot(self, save=False, savepath = "pcs.pdf"):
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.features)
        
        colors = ["tab:blue", "tab:orange"]
        # Plot the 2D representation
        #c=list(map(str, self.classes)), label=list(map(str, self.classes)), cmap=matplotlib.colors.ListedColormap(colors)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.classes)
        plt.title(f'2D Projection of {self.feature_space}D Data using PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        #plt.savefig("pca.pdf")
        plt.legend(*scatter.legend_elements())
        if save:
            print("figure saved")
            plt.tight_layout()
            plt.savefig(savepath)
        plt.show()


    def tSNE(self, savefig = False, savepath = "tsne.pdf"):
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(self.features)
        scatter = plt.scatter(X_embedded[:,0],X_embedded[:,1],c=self.classes)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(*scatter.legend_elements())
        if savefig:
            print("figure saved")
            plt.tight_layout()
            plt.savefig(savepath)
        plt.show()
        return
#%%
if __name__ == "__main__":
    print("labinium.py imported")
    #cutoff = [40e6, 60e6, 80e6, 100e6, 150e6, 200e6, 250e6]
    #features = [50, 100, 200, 500, 1000, 2000, 5000, 7500, 10000]
    #struct = [(30), (20, 10), (10, 20), (10, 10, 10), (5, 10, 15), (15, 10, 5), (50), (100), (50, 50)]

    cutoff = [40e6]
    features = [50]
    struct = [(30)]
    
    
    results = np.zeros((len(cutoff), len(features), len(struct)))

    for x, c in tqdm(enumerate(cutoff)):
        for y, f in enumerate(features):
            a = speechClassifierLab("../Lab/Classification/FullRunBin/", cutoff=c, feature_space=f)
            for z, i in enumerate(struct):
                print(f"Cutoff: {c}, Features: {f}, Structure: {i}")
                results[x,y,z] = a.clf(layers = i)
                print("PCA")
                a.PCA_plot(save=True)
                print("tSNE")
                a.tSNE(savefig=True)
                print("\n\n")

# %%



# %%
