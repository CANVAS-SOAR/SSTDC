import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from matplotlib.pyplot import specgram
from scipy import signal


path = '../data/5_22/datalogs/'	# change to fit your data loc
fname = path + 'rad_2017-05-22_18_39_25.dat'			# filename
'''
path = '/home/ryan/Documents/SSTDC/data/5_21/datalogs/'	# change to fit your data loc
fname = path + 'rad_2017-05-21_18:03:48.dat'			# filename
'''
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
	#f, t, Sxx = signal.spectrogram(array, fs=FS, window=signal.hamming(10000,sym=True), noverlap=9000)	# input array and sample rate, use hamming window with 90% overlap
	f, t, Sxx = signal.spectrogram(array, fs=FS, scaling='density', window=signal.hamming(10000,sym=True), noverlap=9000)
	#fig = plt.subplot(1, 1, 1)
	print(Sxx)

	#plt.pcolormesh(t, f, Sxx, norm=colors.PowerNorm(gamma=1./12.), cmap='gist_heat')
	#plt.pcolormesh(t, f, Sxx, norm=colors.PowerNorm(gamma=1./16.), cmap='jet')
	#plt.pcolormesh(t, f, Sxx, norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='gist_heat')
	plt.pcolormesh(t, f, Sxx, norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='jet')	
	#plt.pcolormesh(t, f, Sxx)

	plt.ylabel('Frequency [Hz]')	# Labels
	plt.xlabel('Time [sec]')
	axes = plt.gca()				# gets the current axis instances
	axes.set_ylim([0,1000])			# arbitrarily capped, otherwise max y is 44100/2
	
	#connection_id = fig.canvas.mpl_connect('button_press_event', clickXY)
	
	plt.show()						# show
	return Sxx

def clickXY(event):
	if event.button:
		print("I clicked")


recording = getNParray(fname)		# read file
spectro = Spec(recording, 44100)	# draw spectrogram


print("End")

