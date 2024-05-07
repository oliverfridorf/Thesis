#%%
#%% Imports

# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Signal processing
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft

# File handling
from scipy.io.wavfile import read, write
import os

# Machine learning
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.manifold import TSNE

# Audio processing
from librosa import effects
import colorednoise as cn



## Function to generate noisy data sample from the FSDD dataset ##
def noisyData(data_path = "FSDD/recordings", output_path = "noise_test", 
              noise_type='gauss', snr=20, pf = 5):
    '''
    Add noise to data
    Noise types: 'gauss', 'pink', 'pitch', 'stretch'
    '''
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"Directory '{output_path}' created successfully.")
    
    
    if noise_type == 'gauss':
        files = os.listdir(data_path)
        print("Loaded", len(files), "files successfully. Adding gaussian noise...")
        for f in files:
            if f.startswith(("0", "1")):
                sr, data = read(f"{data_path}/{f}")
                amp_sig = np.max(np.abs(data))         # To define signal to noise
                amp_noi = amp_sig / (10**(snr/20))  # Amplitude of noise
                noise = np.random.normal(0, 1, len(data))*amp_noi
                
                data_noisy = data + noise 
                data_noisy = data_noisy.astype(np.int16)
                # Save noisy data
                write(f"{output_path}/{f}", sr, data_noisy)
        print(f"Noisy gaussian data saved in {output_path}\n")
    
    elif noise_type == 'pink':
        files = os.listdir(data_path)
        print("Loaded", len(files), "files successfully. Adding pink noise...")
        for f in files:
            if f.startswith(("0", "1")):
                sr, data = read(f"{data_path}/{f}")
                amp_sig = np.max(np.abs(data))         # To define signal to noise
                amp_noi = amp_sig / (10**(snr/20))  # Amplitude of noise
                noise = cn.powerlaw_psd_gaussian(1, len(data))*amp_noi
                
                data_noisy = data + noise 
                data_noisy = data_noisy.astype(np.int16)
                # Save noisy data
                write(f"{output_path}/{f}", sr, data_noisy)
        print(f"Noisy pink data saved in {output_path}\n")

    elif noise_type == 'pitch':
        files = os.listdir(data_path)
        print("Loaded", len(files), "files successfully. Adding pitch shift...")
        for f in files:
            if f.startswith(("0", "1")):
                sr, data = read(f"{data_path}/{f}")
                step = np.random.randint(-pf, pf)
                data = data.astype(np.float32)
                data_noisy = effects.pitch_shift(y=data, sr=sr, n_steps=step, n_fft=1024)
                data_noisy = data_noisy.astype(np.int16)      
                write(f"{output_path}/{f}", sr, data_noisy)
        print(f"Noisy pitch shift data saved in {output_path}\n")
    else:
        print("Invalid noise type. Choose from 'gauss', 'pink', 'pitch'")         
                
    return

## Class to setup and generate the files needed for simulations ##
class simSetup:
    def __init__(self, mumax_sim_folder, signal_folder = "FSDD/recordings"):
        
        self.mumax_sim_folder = mumax_sim_folder
        self.signal_folder = signal_folder


    def create_sim_files(self,
        max_T,
        runtime,
        digits = ("0","1"),
        #input_folder = "FSDD/recordings",
        #output_folder = "Sim_files",
        ):

        print("Creating simulation files")

        output_folder = self.mumax_sim_folder
        input_folder = self.signal_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Directory '{output_folder}' created successfully.")
        else:
            print(f"Directory '{output_folder}' already exists.")

        files = os.listdir(input_folder)
        for f in files:
            if f.startswith(digits):
                record = f
                spoken_digit_read = read(f"{input_folder}/{record}")[1] # get array not samplingrate
                spoken_digit = np.copy(spoken_digit_read).astype("float")
            
                N = len(spoken_digit)
            
                # create linspace with runtime and timesteps of dt
                t, dt = np.linspace(0,runtime,N, retstep=True)

                # scaling
                spoken_digit *= max_T/np.max(np.abs(spoken_digit))
                
                # read setup file
                with open("speech_setup.txt") as f:
                    setup_file = f.read() # start from scratch

                # setup savetime of dt
                setup_file += (f'\ntableautosave({dt})')

                # setup applied b field and runtime
                for b in spoken_digit:
                    setup_file += (f"\nB_ext = vector({b},0,0)")
                    setup_file += (f"\nrun({dt})")

                # set output formaet as .txt
                record = record.replace(".wav",".txt")
                # save file
                with open(f"{output_folder}/{record}", "w") as f:
                    f.write(setup_file)
        
            else:
                break


    
    def que_sims(self, shuffle = False):
        '''run simulations from folder'''
        print(f'Queing {len(self.mumax_sim_folder)} simulations')

        files = os.listdir(self.mumax_sim_folder)

        if shuffle:
            np.random.shuffle(files)

        for f in (files):
            print(f'running {f}')
            os.system(f"mumax3 {self.mumax_sim_folder}/{f}")


## Class to plot the results from a simulation ##
class simPlotter:
    def __init__(self, path, save_figures = False):

        self.path = path
        print(f'Getting data from {self.path}')

        self.save_figures = save_figures

    

    def read_data(self):
        #reading data
        print(f"Reading data from {self.path}")
        df = pd.read_table(self.path)
        #print(df)
        t = df.loc[:,"# t (s)"].to_numpy()
        my = df.loc[:,"my ()"].to_numpy()
        bext = df.loc[:,"B_extx (T)"].to_numpy()
        return t, my, bext
            
    def B_ext(self, save_only = False):
        print("plottting B_ext")
        t, _, bext = self.read_data()
        plt.plot(t*1e9, bext*1e3)
        plt.xlabel("Time [ns]")
        plt.ylabel("B-Field [mT]")
        if self.save_figures or save_only:
            plt.savefig("Bext.pdf")
        plt.show()
    
    def my(self):
        print("plottting my")
        t, my, _ = self.read_data()
        plt.plot(t*1e9, my)
        plt.ylabel("Magnetization")
        plt.xlabel("Time [ns]")
        if self.save_figures:
            plt.savefig("my.pdf")
        plt.show()
    
    def STFT(self, ylim = None, feature_space = 30, log = False):
        print("plottting STFT")
        t, my, _ = self.read_data()

        dt = t[1]-t[0]
        N = len(t)

        f, t, z = stft(my, fs = 1/dt, nperseg=int(N/feature_space))
        z = np.abs(z)
        if log:
            z = 20*np.log10(z)
        plt.pcolormesh(t*1e9, f/1e9, (z))
        if ylim != None:
            plt.ylim(0,ylim) # in GHz
        plt.xlabel("Time [ns]")
        plt.ylabel("Frequency [GHz]")
        plt.colorbar()
        #plt.savefig("ex_mag_stft.pdf")
        if self.save_figures:
            plt.savefig("../Dropbox/Apps/Overleaf/Speciale/Chapter3_simulation/fig/STFT.pdf")
        plt.show()
    
    def all(self):
        self.B_ext()
        self.my()
        self.STFT()


## Function to find the maximum frequency of a signal ##
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


## Class to setup and train classifier on simulation data ##
class speechClassifierSimulation():
    def __init__(self, data_folder, feature_space = 20, cutoff = 200e6, processing = True):
        
        
        self.feature_space = feature_space
        print('Preparing data')
        # loading data
        files = []
        # get .out files
        for file in os.listdir(data_folder):
                if file.endswith(".out"):
                    files.append(file)
        
        self.features = np.zeros((len(files), feature_space))
        self.classes = []

        for n, file in (enumerate(files)):
            try:
                #print(file)
                # load data
                df = pd.read_table(f'{data_folder}/{file}/table.txt')

                # append class from filename
                self.classes.append(int(file[0]))

                # get data from 
                if processing:
                    my = df.loc[:,"my ()"].to_numpy()
                else:
                    my = df.loc[:,"B_extx (T)"].to_numpy()

                t = df.loc[:,"# t (s)"].to_numpy()

                for i,(ti,myi) in enumerate(zip(np.array_split(t,feature_space), np.array_split(my,feature_space))):
                    
                    self.features[n,i] = get_f_max(ti, myi, cutoff=cutoff)
            except:
                #files.pop(n)
                print(f'{file} pooped')
                #self.classes.append("-1")
                #self.features.reshape((len(files)-1),feature_space)
            
        print("Done")


    def PCA_plot(self):
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
        plt.show()


    def tSNE(self, savefig = False, savepath = "../Dropbox/Apps/Overleaf/Speciale/Chapter3_simulation/fig/tsne.pdf"):
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

