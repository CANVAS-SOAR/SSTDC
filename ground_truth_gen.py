import sys, os
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.pyplot import specgram
from scipy import signal


path = '/home/ryan/Documents/SSTDC/data/5_18/datalogs/'	# change to fit your data loc
fname = path + 'rad_2017-05-18_22:20:32.dat'			# filename

'''
Name = getNParray
Purpose = read radar recording .dat files
Parameters = filename = name of file to turn to np array
returns = np array of recording
** Note: only tested and works on 1 channel, requires mods for 2 channel **
'''
def getNParray(filename):

	data = open(filename,"r")	# open file
	rec_data = np.array([])		# store the recorded data
	buf = np.zeros(4000)		# temp buffer, prevents copy full np array ever value
	index = 0					# track where to insert into temp buf
	for line in data:			# for everything in the line of text in .dat
		if( line.find('[') != -1 ):	# if start of chunk
			line = line.replace('[', '')	# remove [
		elif( line.find(']') != -1):		# if it is end of chunk remove ]
			line = line.replace(']', '')
			line_split = line.split()		# split on space
			for value in line_split:		# for each value
				buf[index] = value			# insert to buf
				index += 1					# increment location	
			rec_data = np.append(rec_data,buf)	# add buf to data only at end of chunk
			buf = np.zeros(4000)			# reset buf and index
			index = 0
		else:
			line_split = line.split()	# not end or beginning, add to buf
			for value in line_split:	# all values in line
				buf[index] = value
				index += 1
	return rec_data			# return recoding in np.array

'''
Name = Spec
Purpose = create spectrogram from np array of radar recording
Parameters = array=np array of recording, FS=sampling rate
returns = the spectrogram intensity map as an np array
'''
def Spec(array, FS):
	f, t, Sxx = signal.spectrogram(array, fs=FS, window=signal.hamming(10000,sym=True), noverlap=9000)	# input array and sample rate, use hamming window with 90% overlap
	plt.pcolormesh(t, f, Sxx)		# I believe for setting the color settings
	plt.ylabel('Frequency [Hz]')	# Labels
	plt.xlabel('Time [sec]')
	axes = plt.gca()				# not sure what this line does
	axes.set_ylim([0,800])			# arbitrarily capped, otherwise max y is 44100/2
	plt.show()						# show
	return Sxx


recording = getNParray(fname)		# read file
spectro = Spec(recording, 44100)	# draw spectrogram


print("End")

