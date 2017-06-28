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

data_save_dir = './saved_data'
utdata_save_file = data_save_dir + '/utdata'
uttarget_save_file = data_save_dir + '/uttarget'
utlabel_save_file = data_save_dir + '/utlabel'
usdata_save_file = data_save_dir + '/usdata'
ustarget_save_file = data_save_dir + '/ustarget'
tdata_save_file = data_save_dir + '/tdata'
ttarget_save_file = data_save_dir + '/ttarget'
tlabel_save_file = data_save_dir + '/tlabel'
sdata_save_file = data_save_dir + '/sdata'
starget_save_file = data_save_dir + '/starget'

if not os.path.exists(data_save_dir):
	os.mkdir(data_save_dir)
	utdata, uttarget, utlabel, usdata, ustarget = get_training_and_testing_data() # get unscaled training and testing data
	tdata, ttarget, tlabel, sdata, starget = scale_data(utdata, uttarget, utlabel, usdata, ustarget) # scale the data
	np.save(utdata_save_file, utdata)
	np.save(uttarget_save_file, uttarget)
	np.save(utlabel_save_file, utlabel)
	np.save(usdata_save_file, usdata)
	np.save(ustarget_save_file, ustarget)
	np.save(tdata_save_file, tdata)
	np.save(ttarget_save_file, ttarget)
	np.save(tlabel_save_file, tlabel)
	np.save(sdata_save_file, sdata)
	np.save(starget_save_file, starget)
else:
	utdata = np.load(utdata_save_file + '.npy')
	uttarget = np.load(uttarget_save_file + '.npy')
	utlabel = np.load(utlabel_save_file + '.npy')
	usdata = np.load(usdata_save_file + '.npy')
	ustarget = np.load(ustarget_save_file + '.npy')
	tdata = np.load(tdata_save_file + '.npy')
	ttarget = np.load(ttarget_save_file + '.npy')
	tlabel = np.load(tlabel_save_file + '.npy')
	sdata = np.load(sdata_save_file + '.npy')
	starget = np.load(starget_save_file + '.npy')

utdata_x = np.delete(utdata, [1, 2], axis=2)								  # track's x axis of unscaled training data shape: (3000, 300, 1)
utdata_t = np.delete(utdata, [0, 1], axis=2)								  # track's t axis of unscaled training data
uttarget_x = np.delete(uttarget, [1], axis=1)								  # target's x axis of unscaled training data
usdata_x = np.delete(usdata, [1, 2], axis=2)								  # track's x axis of unscaled testing data
usdata_t = np.delete(usdata, [0, 1], axis=2)								  # track's t axis of unscaled testing data
ustarget_x = np.delete(ustarget, [1], axis=1)								  # target's x axis of unscaled testing data
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

utdata_delta_x = np.max(utdata_x, axis=1) - np.min(utdata_x, axis=1)
utdata_delta_t = np.max(utdata_t, axis=1) - np.min(utdata_t, axis=1)
avg_velocity_t = np.divide(utdata_delta_x,utdata_delta_t) # be care of overflow
tfeature[:,0]= avg_velocity_t.reshape((1,3000))

usdata_delta_x = np.max(usdata_x, axis=1) - np.min(usdata_x, axis=1)
usdata_delta_t = np.max(usdata_t, axis=1) - np.min(usdata_t, axis=1)
avg_velocity_s = np.divide(usdata_delta_x,usdata_delta_t) # be care of overflow
sfeature[:,0]= avg_velocity_s.reshape((1,100000))

# 2 1. x.end - x0

'''
utdata_x_trans=utdata_x[:,:,0]                                                #target's x axis endpoint in shape (3000,300)
endpoints=[]
for row in utdata_x_trans:
	endpoints.append(row[np.nonzero(row)[-1][-1]])							  
endpoint_x_t = np.array(endpoints)                                            #this shall result in a (1,300) array with the last nonzero index
endpoint_target_distance=endpoint_x_t-uttarget_x[:,0]               		  #fill the tfeature with unscaled 
endpoint_target_distance=endpoint_target_distance/(np.max(endpoint_x_t)-np.min(endpoint_x_t))
tfeature[:,1]=endpoint_target_distatnce
'''

tdata_x_trans=tdata_x[:,:,0]
endpoint_t=count_record_num("t")-1
tfeature[:,1]=tdata_x_trans[range(3000),endpoint_t]-ttarget_x[:,0]

sdata_x_trans=sdata_x[:,:,0]
endpoint_s=count_record_num("s")-1
sfeature[:,1]=sdata_x_trans[range(100000),endpoint_s]-starget_x[:,0]		  #the result is scaled


# 3 2.x.max-x.min

tdata_delta_x = np.max(utdata_x, axis=1) - np.min(utdata_x, axis=1)
tfeature[:, 2] = tdata_delta_x.reshape((1, 3000))

sdata_delta_x = np.max(usdata_x, axis=1) - np.min(usdata_x, axis=1)
sfeature[:, 2] = sdata_delta_x.reshape((1, 100000))

# 4 3. mean of sigma|x-x0|^2
count_record_tnum = count_record_num(training_or_testing="t")
for i in range(3000):
	sigma_ = 0
	for j in range(count_record_tnum[i]-1):
		#set x0 to 0 when no data, set x0 to target_x when there is data
		sigma_ = sigma_ + abs(tdata_x[i, j, 0] - ttarget_x[i,0])**2 # no squeeze
	tfeature[i, 3] = sigma_/count_record_tnum[i] #mean of sigma

count_record_snum = count_record_num(training_or_testing="s")
for i in range(100000):
	sigma_ = 0
	for j in range(count_record_snum[i]-1):
		#set x0 to 0 when no data, set x0 to target_x when there is data
		sigma_ = sigma_ + abs(sdata_x[i, j, 0] - starget_x[i,0])**2 # no squeeze
	sfeature[i, 3] = sigma_/count_record_snum[i] #mean of sigma


# 5 13 a new feature, depicting how many same time-point that a data possess
utdata_x_diff=np.diff(utdata_x[:,:,0])										  #np.diff calculate the adjacent difference
utdata_t_diff=np.diff(utdata_t[:,:,0])
velocity_t=utdata_x_diff/utdata_t_diff										  #result in a number of nan
number_of_same_time_t=np.sum((np.isinf(velocity_t)),axis=1)					  # a (3000,) data max is 144, min if 0 there are 574 entries that has same t in total
#index_of_same_time_t=np.where(number_of_same_time_t>0)
tfeature[:,13]=number_of_same_time_t

usdata_x_diff=np.diff(usdata_x[:,:,0])										  #np.diff calculate the adjacent difference
usdata_t_diff=np.diff(usdata_t[:,:,0])
velocity_s=usdata_x_diff/usdata_t_diff										  #result in a number of nan
number_of_same_time_s=np.sum((np.isinf(velocity_s)),axis=1)					  # a (10000,) data max is 241, min if 0 there are 14777 entries that has same t in total
#index_of_same_time_t=np.where(number_of_same_time_t>0)
sfeature[:,13]=number_of_same_time_s

# 5 14 the mean of velocity

velocity_t_copy=np.copy(velocity_t)											
velocity_t_copy[np.isinf(velocity_t_copy)]=300								  #stipulate that the speed of inf equals 300 for inf
velocity_t_copy[velocity_t_copy>300]=300
velocity_t_mean=np.nanmean(np.absolute(velocity_t_copy),axis=1)				  #no scaling because velocity is too big for some, resulting in infinitisimal for the normal one
tfeature[:,14]=velocity_t_mean												  #/(np.amax(velocity_t_mean)-np.amin(velocity_t_mean)) this i for scaling

velocity_s_copy=np.copy(velocity_s)
velocity_s_copy[np.isinf(velocity_s_copy)]=300								  #stipulate that the speed of inf equals 300 for inf
velocity_s_copy[velocity_s_copy>300]=300									  #a strange case with max 136900 is discovered, probably the t is too small
velocity_s_mean=np.nanmean(np.absolute(velocity_s_copy),axis=1)				  #no scaling because velocity is too big for some, resulting in infinitisimal for the normal one
sfeature[:,14]=velocity_s_mean												  #/(np.amax(velocity_s_mean)-np.amin(velocity_s_mean)) this is for scaling

# 5 4. the standard deviation of velocity

velocity_t_std=np.nanstd(velocity_t_copy,axis=1)							  #calculate the root mean square of the velocity
tfeature[:,4]=velocity_t_std												  #unscaled because the velocity difference is too big, and it is standard deviation anyway

velocity_s_std=np.nanstd(velocity_s_copy,axis=1)
sfeature[:,4]=velocity_s_std


# 5 15.|xi-xi+1|/|ti-ti+1|^2 mean of acceleration

acceleration_t=velocity_t/utdata_t_diff
acceleration_t[np.isinf(acceleration_t)]=50
acceleration_t[acceleration_t>100]=50
tfeature[:,5]=np.nanmean(np.absolute(acceleration_t),axis=1)

acceleration_s=velocity_s/usdata_t_diff
acceleration_s[np.isinf(acceleration_s)]=50
acceleration_s[acceleration_s>100]=50
sfeature[:,15]=np.nanmean(np.absolute(acceleration_s),axis=1)

#5 5. the standard deviation of accelration
tfeature[:,5]=np.nanstd(acceleration_t,axis=1)
sfeature[:,5]=np.nanstd(acceleration_s,axis=1)

# 6 6.Stop times:
tdata_x_no = np.squeeze(tdata_x, axis=2)
tdata_x_no1 = np.delete(tdata_x_no, [0], axis=1) # shape: (3000, 299)
tdata_x_non = np.delete(tdata_x_no, [299], axis=1)
tdata_t_no = np.squeeze(tdata_t, axis=2)
tdata_t_no1 = np.delete(tdata_t_no, [0], axis=1)
tdata_t_non = np.delete(tdata_t_no, [299], axis=1)
tdata_k = np.abs(np.divide(tdata_x_no1 - tdata_x_non, tdata_t_no1 - tdata_t_non))
threshold = 0.1
record_num = count_record_num(training_or_testing="t")
for i in range(3000):
	tdata_k_tmp = tdata_k[i]
	gt_threshold = 0
	lt_threshold = 0
	stop_begin_point = 0
	for j in range(record_num[i] - 1):
		if j == 298:
			break
		if tdata_k_tmp[j] > threshold and tdata_k_tmp[j+1] < threshold:
			gt_threshold += 1
			stop_begin_point = j
		elif tdata_k_tmp[j] < threshold and tdata_k_tmp[j+1] > threshold:
			if j == stop_begin_point + 1 and gt_threshold != 0:
				gt_threshold -= 1
			lt_threshold += 1
	tfeature[i, 6] = min(gt_threshold, lt_threshold)

sdata_x_no = np.squeeze(sdata_x, axis=2)
sdata_x_no1 = np.delete(sdata_x_no, [0], axis=1) # shape: (100000, 299)
sdata_x_non = np.delete(sdata_x_no, [299], axis=1)
sdata_t_no = np.squeeze(sdata_t, axis=2)
sdata_t_no1 = np.delete(sdata_t_no, [0], axis=1)
sdata_t_non = np.delete(sdata_t_no, [299], axis=1)
sdata_k = np.abs(np.divide(sdata_x_no1 - sdata_x_non, sdata_t_no1 - sdata_t_non))
threshold = 0.1
record_num = count_record_num(training_or_testing="s")
for i in range(100000):
	sdata_k_tmp = sdata_k[i]
	gt_threshold = 0
	lt_threshold = 0
	stop_begin_point = 0
	for j in range(record_num[i] - 1):
		if j == 298:
			break
		if sdata_k_tmp[j] > threshold and sdata_k_tmp[j+1] < threshold:
			gt_threshold += 1
			stop_begin_point = j
		elif sdata_k_tmp[j] < threshold and sdata_k_tmp[j+1] > threshold:
			if j == stop_begin_point + 1 and gt_threshold != 0:
				gt_threshold -= 1
			lt_threshold += 1
	sfeature[i, 6] = min(gt_threshold, lt_threshold)


# 7 7.Any fluctuation when stopping The continuous 5 x_diff is 0: mean ratio of no_fluct_num++ OR fluctuation of slope
record_num_t = count_record_num(training_or_testing="t")
tdata_x_diff=np.diff(tdata_x[:,:,0])
tdata_t_diff=np.diff(tdata_t[:,:,0])
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
sdata_x_diff=np.diff(sdata_x[:,:,0])
sdata_t_diff=np.diff(sdata_t[:,:,0])
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

# 8 8.Back distance:
utdata_x_trans=utdata_x[:,:,0]
utdata_x_route_length=np.sum(np.absolute(utdata_x_trans),1)                         #the overall route length of the mouse
utdata_x_diff_copy=np.diff(utdata_x_trans).copy()
utdata_x_diff_copy[utdata_x_diff_copy>0]=0
tfeature[:,8]=(-np.sum(utdata_x_diff_copy,1))/utdata_x_route_length					#the portion that backward route take up			

usdata_x_trans=usdata_x[:,:,0]
usdata_x_route_length=np.sum(np.absolute(usdata_x_trans),1)							#the overall route for the test data
usdata_x_diff_copy=np.diff(usdata_x_trans).copy()
usdata_x_diff_copy[usdata_x_diff_copy>0]=0
sfeature[:,8]=(-np.sum(usdata_x_diff_copy,1))/usdata_x_route_length					#the whole thing is positive and naturally scaled

# 9 9.
# 9 10.
'''
utdata_xs = np.squeeze(utdata_x, axis=2)
utdata_ts = np.squeeze(utdata_t, axis=2)
utdata_xss = np.zeros((3000, 300))
for i in range(3000):
	num_of_nonzeros = np.count_nonzero(utdata_xs[i])
	window_size = int(np.floor(num_of_nonzeros / 4 + 1))
	if num_of_nonzeros > 100 and num_of_nonzeros <= 200:
		poly_size = int(np.floor(num_of_nonzeros / 20))
	elif num_of_nonzeros > 200:
		poly_size = int(np.floor(num_of_nonzeros / 25))
	else:
		poly_size = int(np.floor(num_of_nonzeros / 20 + 5))
	utdata_xss[i, 0: num_of_nonzeros] = savitzky_golay(utdata_xs[i, 0: num_of_nonzeros], window_size, poly_size)
'''

utdata_s = np.copy(utdata)
for i in range(3000):
	num_of_nonzeros = np.count_nonzero(utdata_s[i, :, 0])
	window_size = int(np.floor(num_of_nonzeros / 4 + 1))
	if num_of_nonzeros > 100 and num_of_nonzeros <= 200:
		poly_size = int(np.floor(num_of_nonzeros / 20))
	elif num_of_nonzeros > 200:
		poly_size = int(np.floor(num_of_nonzeros / 25))
	else:
		poly_size = int(np.floor(num_of_nonzeros / 20 + 5))
	utdata_s[i, 0: num_of_nonzeros, 0] = savitzky_golay(utdata_s[i, 0: num_of_nonzeros, 0], window_size, poly_size)
tdata_s, _ = scale_one_data(utdata_s)
tsmooth_x = np.abs(tdata_s[:, :, 0] - tdata[:, :, 0])
tsmooth_x_mse = (tsmooth_x ** 2).mean(axis=1) # SOME NAN AFTER THIS STEP!!!!
											# SEE LATER IF NEEDS FIXED
											# np.count_nonzero(~np.isnan((tsmooth_x_mse))) --> NON-NAN: 2924
tfeature[:, 9] = np.mean(tsmooth_x, axis=1).reshape((1, 3000))
tfeature[:, 10] = tsmooth_x_mse.reshape((1, 3000))

usdata_s = np.copy(usdata)
for i in range(100000):
	num_of_nonzeros = np.count_nonzero(usdata_s[i, :, 0])
	window_size = int(np.floor(num_of_nonzeros / 4 + 1))
	if num_of_nonzeros > 100 and num_of_nonzeros <= 200:
		poly_size = int(np.floor(num_of_nonzeros / 20))
	elif num_of_nonzeros > 200:
		poly_size = int(np.floor(num_of_nonzeros / 25))
	else:
		poly_size = int(np.floor(num_of_nonzeros / 20 + 5))
	usdata_s[i, 0: num_of_nonzeros, 0] = savitzky_golay(usdata_s[i, 0: num_of_nonzeros, 0], window_size, poly_size)
sdata_s, _ = scale_one_data(usdata_s)
ssmooth_x = np.abs(sdata_s[:, :, 0] - sdata[:, :, 0])
ssmooth_x_mse = (ssmooth_x ** 2).mean(axis=1) # SOME NAN AFTER THIS STEP!!!!
											# SEE LATER IF NEEDS FIXED
											# np.count_nonzero(~np.isnan((ssmooth_x_mse))) --> NON-NAN: 2924
sfeature[:, 9] = np.mean(ssmooth_x, axis=1).reshape((1, 100000))
sfeature[:, 10] = ssmooth_x_mse.reshape((1, 100000))

# 10 11.if x<x0, sum of all t_diff when x unchange - diff && count_inf OR 0/1 mask
record_num_t = count_record_num(training_or_testing="t")
utdata_x_diff=np.diff(utdata_x[:,:,0])
utdata_t_diff=np.diff(utdata_t[:,:,0])
x_unchange_flag_t = np.ones(utdata_x_diff.shape)/utdata_x_diff
bool_same_x_t = np.isinf(x_unchange_flag_t) # inf convert to 1, otherwise 0
#number_of_same_x_t=np.sum((np.isinf(x_unchange_flag)),axis=1)
for i in range(3000):
	sum_time_unchange = 0
	for j in range(record_num_t[i]-2): # shape deducted one after diff
		if bool_same_x_t[i,j]==1 and utdata_x[i,j,0] < uttarget_x[i,0]: #x<x0
			sum_time_unchange = sum_time_unchange + utdata_t_diff[i,j]
	tfeature[i,11] = sum_time_unchange

record_num_s = count_record_num(training_or_testing="s")
usdata_x_diff=np.diff(usdata_x[:,:,0])
usdata_t_diff=np.diff(usdata_t[:,:,0])
x_unchange_flag_s = np.ones(usdata_x_diff.shape)/usdata_x_diff
bool_same_x_s = np.isinf(x_unchange_flag_s) # inf convert to 1, otherwise 0
#number_of_same_x_t=np.sum((np.isinf(x_unchange_flag)),axis=1)
for i in range(100000):
	sum_time_unchange = 0
	for j in range(record_num_s[i]-2):
		if bool_same_x_s[i,j]==1 and usdata_x[i,j,0] < ustarget_x[i,0]:
			sum_time_unchange = sum_time_unchange + usdata_t_diff[i,j]
	sfeature[i,11] = sum_time_unchange

# 11 12. judging the similarity of the line with a straight line
#        there are two cases, one with a direct line, another with x remaining the same first and then followed by a direct sloping line
#        Then it is natural to focus on the regression of the direct line only.

count_last_nonezero_t=count_record_num("t")-1
utdata_t_trans=utdata_t[:,:,0]
utdata_x_trans=utdata_x[:,:,0]
utdata_diff_x=utdata_x_trans[:,0]-utdata_x_trans[range(3000),count_last_nonezero_t]
utdata_diff_t=utdata_t_trans[:,0]-utdata_t_trans[range(3000),count_last_nonezero_t]
utdata_initial_end_k=utdata_diff_x/utdata_diff_t

utdata_k_diff_sum=np.nansum(np.absolute(tdata_k-np.asarray(utdata_initial_end_k).reshape(3000,1)),axis=1)
utdata_k_diff=np.sqrt(np.nan_to_num(utdata_k_diff_sum))
utdata_k_diff[utdata_k_diff>3000]=3000
tfeature[:,12]=utdata_k_diff   #since k can be very large, so we use the square root to decrease the difference

count_last_nonezero_s=count_record_num("s")-1
usdata_t_trans=usdata_t[:,:,0]
usdata_x_trans=usdata_x[:,:,0]
usdata_diff_x=usdata_x_trans[:,0]-usdata_x_trans[range(100000),count_last_nonezero_s]
usdata_diff_t=usdata_t_trans[:,0]-usdata_t_trans[range(100000),count_last_nonezero_s]
usdata_initial_end_k=usdata_diff_x/usdata_diff_t

usdate_k_diff_sum=np.nansum(np.absolute(sdata_k-np.asarray(usdata_initial_end_k).reshape(100000,1)),axis=1)
usdata_k_diff=np.sqrt(np.nan_to_num(usdate_k_diff_sum))
usdata_k_diff[usdata_k_diff>3000]=3000
sfeature[:,12]=usdata_k_diff

if not os.path.exists(var_save_dir):
	os.mkdir(var_save_dir)

np.save(tfeature_save_file, tfeature)
np.save(sfeature_save_file, sfeature)
np.save(tlabel_save_file, tlabel)

