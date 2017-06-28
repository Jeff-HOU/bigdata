import numpy as np
import os
from functions import get_training_and_testing_data, scale_data, scale_one_data, \
					  savitzky_golay, count_record_num

# training_file = '../data/dsjtzs_txfz_training.txt'
# testing_file = '../data/dsjtzs_txfz_test1.txt'

var_save_dir = './saved_vars'
tfeature_save_file = var_save_dir + '/tfeature'
sfeature_save_file = var_save_dir + '/sfeature'
tlabel_save_file = var_save_dir + '/tlabel'

tfeature = np.zeros((3000, 16)) # feature array of training data
sfeature = np.zeros((100000, 16)) # feature array of testing data

# 0.(normalization)
utdata, uttarget, utlabel, usdata, ustarget = get_training_and_testing_data() # get unscaled training and testing data
utdata_x = np.delete(utdata, [1, 2], axis=2)								  # track's x axis of unscaled training data shape: (3000, 300, 1)
utdata_t = np.delete(utdata, [0, 1], axis=2)								  # track's t axis of unscaled training data
uttarget_x = np.delete(uttarget, [1], axis=1)								  # target's x axis of unscaled training data
usdata_x = np.delete(usdata, [1, 2], axis=2)								  # track's x axis of unscaled testing data
usdata_t = np.delete(usdata, [0, 1], axis=2)								  # track's t axis of unscaled testing data
ustarget_x = np.delete(ustarget, [1], axis=1)								  # target's x axis of unscaled testing data
tdata, ttarget, tlabel, sdata, starget = scale_data(utdata, uttarget, utlabel, usdata, ustarget) # scale the data
tdata_x = np.delete(tdata, [1, 2], axis=2)									  # track's x axis of scaled training data
tdata_t = np.delete(tdata, [0, 1], axis=2)									  # track's t axis of scaled training data
ttarget_x = np.delete(ttarget, [1], axis=1)									  # target's x axis of scaled training data
sdata_x = np.delete(sdata, [1, 2], axis=2)									  # track's x axis of scaled testing data
sdata_t = np.delete(sdata, [0, 1], axis=2)									  # track's t axis of scaled testing data
starget_x = np.delete(starget, [1], axis=1)									  # target's x axis of scaled testing data
#-----------------------------------------
# task_id index_in_tfeature. task_name  ||
# |  _________|    ______________|      ||
# | |             |                     ||
# 1 0.(x.max-x.min)/(t.max-t.min)
## TODO: WHAT IS THIS????
'''
utdata_delta_x = np.max(utdata_x, axis=1) - np.min(utdata_x, axis=1)
utdata_delta_t = np.max(utdata_t, axis=1) - np.min(utdata_t, axis=1)
avg_velocity_t = np.divide(utdata_delta_x,utdata_delta_t) # be care of overflow
tfeature[:,0]= avg_velocity_t.reshape((1,3000))

usdata_delta_x = np.max(usdata_x, axis=1) - np.min(usdata_x, axis=1)
usdata_delta_t = np.max(usdata_t, axis=1) - np.min(usdata_t, axis=1)
avg_velocity_s = np.divide(usdata_delta_x,usdata_delta_t) # be care of overflow
sfeature[:,0]= avg_velocity_s.reshape((1,100000))
'''
# 7 7.Any fluctuation when stopping The continuous 5 x_diff is 0: mean ratio of no_fluct_num++ OR fluctuation of slope
record_num_t = count_record_num(training_or_testing="t")
tdata_x_diff=np.diff(utdata_x[:,:,0])
tdata_t_diff=np.diff(utdata_t[:,:,0])
tdata_k_initial = tdata_x_diff/tdata_t_diff
tdata_k = np.nan_to_num(tdata_k_initial) #change nan to zero(x and t not change so treat it as no fluctuation)
for i in range(3000):
	no_fluct_num_t = 0
	if record_num_t[i] < 5: #avoid NaN
		tfeature[i,7] = 0
		continue
	for j in range(np.floor(record_num_t[i]/5).astype(int)):
		if (abs(tdata_k[i,j]) + abs(tdata_k[i,j+1]) + abs(tdata_k[i,j+2]) + abs(tdata_k[i,j+3]) + abs(tdata_k[i,j+4])) == 0 : #all 0 - the threshold waits for tuning
			no_fluct_num_t = no_fluct_num_t + 1
	tfeature[i,7] = no_fluct_num_t/(record_num_t[i]/5)

record_num_s = count_record_num(training_or_testing="s")
sdata_x_diff=np.diff(usdata_x[:,:,0])
sdata_t_diff=np.diff(usdata_t[:,:,0])
sdata_k_initial = sdata_x_diff/sdata_t_diff
sdata_k = np.nan_to_num(sdata_k_initial) #change nan to zero(x and t not change so treat it as no fluctuation)
for i in range(100000):
	no_fluct_num_s = 0
	if record_num_s[i] < 5: #avoid NaN
		sfeature[i,7] = 0
		continue
	for j in range(np.floor(record_num_s[i]/5).astype(int)):
		if (abs(sdata_k[i,j]) + abs(sdata_k[i,j+1]) + abs(sdata_k[i,j+2]) + abs(sdata_k[i,j+3]) + abs(sdata_k[i,j+4])) == 0 : #all 0 - the threshold waits for tuning
			no_fluct_num_s = no_fluct_num_s + 1
	sfeature[i,7] = no_fluct_num_s/(record_num_s[i]/5)


#record_num_t = count_record_num(training_or_testing="t")
#tdata_x_diff=np.diff(tdata_x[:,:,0])
#tdata_t_diff=np.diff(tdata_t[:,:,0])
#for i in range(3000):
#	no_fluct_num_t = 0
#    if record_num_t[i] < 5: #avoid NaN
#        tfeature[i,7] = 0
#        continue
#	for j in range(record_num_t[i]-5):
#        if (abs(tdata_x_diff[i,j]) + abs(tdata_x_diff[i,j+1]) + abs(tdata_x_diff[i,j+2]) + abs(tdata_x_diff[i,j+3]) + abs(tdata_x_diff[i,j+4])) == 0 : #all 0 - the threshold waits for tuning
#			no_fluct_num_t = no_fluct_num_t + 1
#	tfeature[i,7] = no_fluct_num_t/(record_num_t[i]-5)
#
#record_num_s = count_record_num(training_or_testing="s")
#sdata_x_diff=np.diff(sdata_x[:,:,0])
#sdata_t_diff=np.diff(sdata_t[:,:,0])
#for i in range(100000):
#	no_fluct_num_s = 0
#    if record_num_s[i] < 5:
#        sfeature[i,7] = 0
#        continue
#	for j in range(record_num_s[i]-5):
#		if (abs(sdata_x_diff[i,j]) + abs(sdata_x_diff[i,j+1]) + abs(sdata_x_diff[i,j+2]) + abs(sdata_x_diff[i,j+3]) + abs(sdata_x_diff[i,j+4])) == 0 : #all 0
#			no_fluct_num_s = no_fluct_num_s + 1
#	sfeature[i,7] = no_fluct_num_s/(record_num_s[i]-5)

if not os.path.exists(var_save_dir):
	os.mkdir(var_save_dir)

np.save(tfeature_save_file, tfeature)
np.save(sfeature_save_file, sfeature)
np.save(tlabel_save_file, tlabel)

