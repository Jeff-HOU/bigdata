import numpy as np
from functions import get_training_and_testing_data, scale_data, scale_one_data, \
					  savitzky_golay, count_record_num

training_file = '../data/dsjtzs_txfz_training.txt'
testing_file = '../data/dsjtzs_txfz_test1.txt'

tfeature = np.zeros((3000, 13)) # feature array of training data
sfeature = np.zeros((100000, 13)) # feature array of testing data

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
# 2 1.终点x离目标x距离

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

# 4 3.sigma|x-x0|^2

# 5 4.|xi-xi+1|/|ti-ti+1|
tdata_x_diff=np.diff(tdata_x[:,:,0])										  #np.diff calculate the adjacent difference
tdata_t_diff=np.diff(tdata_t[:,:,0])
velocity_t=tdata_x_diff/tdata_t_diff										  #result in a number of nan
tfeature[:,4]=np.sqrt(np.nanmean(velocity_t**2,axis=1))						  #calculate the root mean square of the velocity, exemting the nan

sdata_x_diff=np.diff(sdata_x[:,:,0])
sdata_t_diff=np.diff(sdata_t[:,:,0])
velocity_s=sdata_x_diff/sdata_t_diff
sfeature[:,4]=np.sqrt(np.nanmean(velocity_s**2,axis=1))


# 5 5.|xi-xi+1|/|ti-ti+1|^2
acceleration_t=velocity_t/tdata_t_diff
tfeature[:,5]=np.sqrt(np.nanmean(acceleration_t**2,axis=1))

acceleration_s=velocity_s/sdata_t_diff
sfeature[:,5]=np.sqrt(np.nanmean(acceleration_s**2,axis=1))					 #follows directly from the above velocity

# 6 6.停的次数:
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


# 7 7.停时有无波动:

# 8 8.折返距离:
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

# 9 9.光滑度
# 9 10.光滑度方差
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
											# np.count_nonzero(~np.isnan((smooth_x_mse)) --> NON-NAN: 2924
tfeature[:, 9] = np.mean(tsmooth_x, axis=1).reshape((1, 3000))
tfeature[:, 10] = tsmooth_x_mse.reshape((1, 3000))

usdata_s = np.copy(usdata)
for i in range(3000):
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
											# np.count_nonzero(~np.isnan((smooth_x_mse)) --> NON-NAN: 2924
sfeature[:, 9] = np.mean(ssmooth_x, axis=1).reshape((1, 100000))
sfeature[:, 10] = ssmooth_x_mse.reshape((1, 100000))

# 10 11.在x<x0时x不变的所有t的总和

# 11 12. judging the similarity of the line with a straight line
count_last_nonezero_t=count_record_num("t")-1
utdata_t_trans=utdata_t[:,:,0]
utdata_diff_x=utdata_x_trans[:,0]-utdata_x_trans[range(3000),count_last_nonezero_t]
utdata_diff_t=utdata_t_trans[:,0]-utdata_t_trans[range(3000),count_last_nonezero_t]
utdata_initial_end_k=utdata_diff_x/utdata_diff_t
utdate_k_diff_sum=np.nansum(np.absolute(tdata_k-np.asarray(utdata_initial_end_k).reshape(3000,1)),axis=1)
tfeature[:,12]=np.sqrt(utdate_k_diff_sum)   #since k can be very large, so we use the square root to decrease the difference

count_last_nonezero_s=count_record_num("s")-1
usdata_t_trans=usdata_t[:,:,0]
usdata_diff_x=usdata_x_trans[:,0]-usdata_x_trans[range(100000),count_last_nonezero_s]
usdata_diff_t=usdata_t_trans[:,0]-usdata_t_trans[range(100000),count_last_nonezero_s]
usdata_initial_end_k=usdata_diff_x/usdata_diff_t
usdate_k_diff_sum=np.nansum(np.absolute(sdata_k-np.asarray(usdata_initial_end_k).reshape(100000,1)),axis=1)
sfeature[:,12]=np.sqrt(usdate_k_diff_sum)