# Get window size
import os, sys
rows, columns = os.popen('stty size', 'r').read().split()

import numpy as np
import pyaudio
import math

from time import localtime, strftime
from scipy import signal

import matplotlib.pyplot as plt

# print entire numpy array
np.set_printoptions(threshold=np.nan)

class SpectrumAnalyzer:
    GAIN = 100
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = int(16000/8)
    START = 0
    N = CHUNK

    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        imic_index = None
        num_devices = self.pa.get_device_count()	# number of devices

        for i in range(num_devices):		# for all devices
            index_full = self.pa.get_device_info_by_index(i)	# get dict info
            name = index_full['name']			# get device name
            if( name.find('iMic') != -1):	# if iMic
                imic_index = index_full['index']# set iMic index
        self.stream = self.pa.open(format = self.FORMAT,
            channels = self.CHANNELS, 
            rate = self.RATE, 
            input = True,
            output = False,
            frames_per_buffer = self.CHUNK,
			input_device_index = imic_index) # set input
        # Main loopbelkin 
        
        self.loop()

    def loop(self):
        try:
            fp = open("datalogs/rad_"+strftime("%Y-%m-%d_%H:%M:%S", localtime())+".dat",'w')
            while True :
                self.data = self.audioinput()
                self.fft()
                #self.spectrogram()
                self.graph()
                #self.graphplot()
                print(self.data, file=fp)

        except KeyboardInterrupt:
            self.pa.close(self.stream)
            fp.close()

        print("End...")

    def audioinput(self):
        ret = self.stream.read(self.CHUNK)#, exception_on_overflow = False)
        ret = np.fromstring(ret, np.float32)
        return ret

    def fft(self):
        self.wave_x = range(self.START, self.START + self.N)
        self.wave_y = self.data[self.START:self.START + self.N]
        self.spec_x = np.fft.fftfreq(self.N, d = 1.0 / self.RATE)  
        y = np.fft.fft(self.data[self.START:self.START + self.N])    
        self.spec_y = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in y]
    
    def spectrogram(self):
        self.f, self.t, self.Sxx = signal.spectrogram(self.spec_x, 10e3)

    def graph(self):
        fft_bin_size = len(self.spec_y)/(int(columns)*2)
        fft_num_bins = int(len(self.spec_y)/fft_bin_size)
        fft_height   = int(rows)

        fft_bins = [np.max(self.spec_y[math.ceil(x*fft_bin_size):math.ceil((x+1)*fft_bin_size)])\
                    for x in range(int(fft_num_bins))]


        os.system('clear')
        for i in range(fft_height):
            for j in range(int(columns)):
                if fft_bins[j] > ((fft_height-i)/fft_height)*(1/self.GAIN)*100:
                    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (i, j, "#"))
                    #print('#', end='')
                else:
                    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (i, j, " "))
                    #print(' ', end='')
        sys.stdout.flush()

        #print(int(np.sum(self.spec_y)/len(self.spec_y)*100)*'=')
        #print(self.spec_y)
    def graphplot(self):
        plt.clf()
        plt.subplot()
        plt.plot(self.spec_x, self.spec_y, linestyle='-')
        plt.axis([0, self.RATE / 2, 0, 25])
        #plt.pcolormesh(self.t, self.f, self.Sxx)
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        plt.pause(.001)

if __name__ == "__main__":
    spec = SpectrumAnalyzer()
