#!/usr/bin/env python
# encoding: utf-8

## Module infomation ###
# Python (3.4.4)
# numpy (1.10.2)
# PyAudio (0.2.9)
# matplotlib (1.5.1)
# All 32bit edition
########################

import numpy as np
import pyaudio
from time import localtime, strftime

import matplotlib.pyplot as plt

class SpectrumAnalyzer:
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 9600
    CHUNK = 1024
    START = 0
    N = 1024

    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = self.FORMAT,
            channels = self.CHANNELS, 
            rate = self.RATE, 
            input = True,
            output = False,
            frames_per_buffer = self.CHUNK)
        # Main loop
        self.loop()

    def loop(self):
        try:
            fp = open("datalogs/rad_"+strftime("%Y%m%d_%H%M%S", localtime())+".dat",'w')
            while True :
                self.data = self.audioinput()
                self.fft()
                self.graphplot()
                repr(self.data)

        except KeyboardInterrupt:
            self.pa.close()
            fp.close()

        print("End...")

    def audioinput(self):
        ret = self.stream.read(self.CHUNK, False)
        ret = np.fromstring(ret, np.float32)
        return ret

    def fft(self):
        self.wave_x = range(self.START, self.START + self.N)
        self.wave_y = self.data[self.START:self.START + self.N]
        self.spec_x = np.fft.fftfreq(self.N, d = 1.0 / self.RATE)  
        y = np.fft.fft(self.data[self.START:self.START + self.N])    
        self.spec_y = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in y]

    def graphplot(self):
        plt.clf()
        plt.subplot()
        plt.plot(self.spec_x, self.spec_y, linestyle='-')
        plt.axis([0, self.RATE / 2, 0, 25])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        plt.pause(.001)

if __name__ == "__main__":
    spec = SpectrumAnalyzer()
