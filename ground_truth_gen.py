import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from matplotlib.pyplot import specgram
from scipy import signal
import time

path = '../data/5_22/datalogs/'	# change to fit your data loc
fname2 = path + 'rad_2017-05-22_22_20_32.dat'			# filename
count = 0
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
def Spec(array, FS, on=True):
	#f, t, Sxx = signal.spectrogram(array, fs=FS, window=signal.hamming(10000,sym=True), noverlap=9000)	# input array and sample rate, use hamming window with 90% overlap
	f, t, Sxx = signal.spectrogram(array, fs=FS, scaling='density', window=signal.hamming(10000,sym=True), noverlap=9000)
	#fig = plt.subplot(1, 1, 1)

	x_bins = np.array([ i for i in range(0,t.shape[0]) ])	# number for each bin in x direction

	#plt.pcolormesh(t, f, Sxx, norm=colors.PowerNorm(gamma=1./12.), cmap='gist_heat')
	#plt.pcolormesh(t, f, Sxx, norm=colors.PowerNorm(gamma=1./16.), cmap='jet')
	#plt.pcolormesh(t, f, Sxx, norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='gist_heat')
	
	# plot vs time
	#plt.pcolormesh(t, f, Sxx, norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='jet')

	# plot vs bin number	
	plt.pcolormesh(x_bins, f, Sxx, norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='jet')

	#plt.pcolormesh(t, f, Sxx)

	plt.ylabel('Frequency [Hz]')	# Labels
	plt.xlabel('Time [sec]')
	axes = plt.gca()				# gets the current axis instances
	axes.set_ylim([0,4000])			# arbitrarily capped, otherwise max y is 44100/2
	
	#connection_id = fig.canvas.mpl_connect('button_press_event', clickXY)
	if( on = True ):
		plt.show(block=False)		# show
	return Sxx


def labelChunk( spec, rec, path, fname, lab, start ):
	new_path = path[:-9]+'labels/'
	name = new_path+fname[:-4]+'_gt.npy'
	if(len(lab) == 0 ):
		lab = np.zeros(spec.shape[1])
	#start = input("Location the start of bin:           ")
	end = input("Location for the end of bin:         ")
	if( end == "end" ):
		end = lab.shape[0]
	obj_class = input("'0-ignore', '1-person', '2-vehicle': ")
	lab[int(start):int(end)] = obj_class
	if( end == lab.shape[0] ):
		np.save(name,lab)
		return name
	else:
		labelChunk( spec, rec, path, fname, lab, end )
		return name 
'''
recording = getNParray(fname)		# read file
spectro = Spec(recording, 44100)	# draw spectrogram
label = np.zeros(spectro.shape[1])	# hold the label of what is in the file
label = labelChunk( label )
print(label[:10])
plt.close()
'''

def checkLabel( lab_name ):
	gt_ary = np.load(label_name)
	change_list = [ 0 ]
	for i in range(gt_ary.shape[0]-1):
		if( gt_ary[i] != gt_ary[i+1] ):
			change_list.append(i)
	change_list.append(gt_ary.shape[0]-1)
	while( len(change_list) > 1 ):
		beg = change_list.pop(0)
		last = change_list[0]
		mid = int((last-beg)/2)+beg
		mid_label = int(gt_ary[mid])
		if(mid_label == 0):
			mid_str = "Nothing"
		elif(mid_label == 1):
			mid_str = "Person"
		elif(mid_label == 2):
			mid_str = "Vehicle"
		print("%d-%d: %s" % (beg, last, mid_str))

for filename in os.listdir(path):
	print("")
	print(count)
	print(filename)
	pass_or_no = input("Pass?")
	if(pass_or_no != "p"):
		label = []
		full_name = path + filename
		recording = getNParray(full_name)
		spectro = Spec(recording, 44100)
		label_name = labelChunk( spectro, recording, path, filename, label, 0 )
		checkLabel( label_name )
		input("Continue to next file? Press any key or quit with control c")
	plt.close('all')
	count += 1

	
print("End")

