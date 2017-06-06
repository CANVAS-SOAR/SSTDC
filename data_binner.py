import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from matplotlib.pyplot import specgram
from scipy import signal

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
	if( on==True):
		plt.show(block=False)		# show
	return Sxx

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


path = '../data/'
net_path = path + 'network_data/'
gt = net_path + 'gt/'
rec = net_path + 'record/'

dirs = [x[0] for x in os.walk(path)]
'''dirs = []

for x in os.walk(path):
	if( x[0].find('labels') != -1 ):
		dirs.append(x)'''

labels = []
datalogs = []
for folder in dirs:
	if( folder.find('labels') != -1 ):
		labels.append(folder)
	elif( folder.find('datalogs') != -1):
		datalogs.append(folder)
print(labels)
print(datalogs)

count_per = 0
count_veh = 0

for fold in labels:
	for label in os.listdir(fold):
		label = fold + '/' + label
		print(label)
		rad_rec = label[:-7] + '.dat'
		rad_rec = rad_rec.replace('labels','datalogs')
		print(rad_rec)
		label_ary = np.load(label) # label np ary
		rad_rec_ary = getNParray(rad_rec) # recording np ary
		rad_rec_spec = Spec(rad_rec_ary, 44100, on=False) # spectrogram

		is_bin = label_ary
		i = 0
		while( i < (label_ary.shape[0]-48) ):
		#for i in range(label_ary.shape[0]-15):
			if( label_ary[i] == 1 ):	# person
				gt_label = np.array([ 1, 0 ])
				if( label_ary[i] == label_ary[i+15] ):
					print("Good bin. store this as np 1")
					i += 48
					if( (count_per % 4) == 0 ):
						gt_loc = gt + 'test/'
						rec_loc = rec + 'test/'
					else:
						gt_loc = gt + 'train/'
						rec_loc = rec + 'train/'
					gt_loc=gt_loc + "gt_%d.npy" % count_per
					rec_loc=rec_loc + "rec_%d.npy" % count_per
					recorded = rad_rec_spec[:794,i:i+48]
					if(recorded.shape == (794, 48)):
						recorded2 = np.zeros((794,48,1))
						for x in [num1 for num1 in range(794)]:
							for y in [num2 for num2 in range(48)]:
								for z in [0]:
									#print("%d %d %d" % (x,y,z))
									recorded2[x,y,z]=recorded[x,y]
						print(gt_loc)
						np.save(gt_loc, gt_label)
						print(rec_loc)
						np.save(rec_loc, recorded2)
						count_per += 1
				else:
					i += 1
			elif( label_ary[i] == 2 ):
				gt_label = np.array([ 0, 1 ])
				if( label_ary[i] == label_ary[i+48] ):
					print("Good bin. store this as np 2")
					i += 48
					if( (count_veh % 4) == 0 ):
						gt_loc = gt + 'test/'
						rec_loc = rec + 'test/'
					else:
						gt_loc = gt + 'train/'
						rec_loc = rec + 'train/'
					gt_loc=gt_loc + "gt_%d.npy" % count_veh
					rec_loc=rec_loc + "rec_%d.npy" % count_veh
					recorded = rad_rec_spec[:794,i:i+48]
					#print(recorded2.shape)
					if(recorded.shape == (794, 48)):
						recorded2 = np.zeros((794,48,1))
						for x in [num1 for num1 in range(794)]:
							for y in [num2 for num2 in range(48)]:
								for z in [0]:
									#print("%d %d %d" % (x,y,z))
									recorded2[x,y,z]=recorded[x,y]
						print(gt_loc)
						np.save(gt_loc, gt_label)
						print(rec_loc)
						np.save(rec_loc, recorded2)
						count_veh += 1
				else:
					i += 1
			else:
				i += 1


		#break

print("End")
print("")
		
