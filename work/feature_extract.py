import numpy as np
from functions import get_training_and_testing_data, scale_data, scale_one_data, \
					  savitzky_golay, count_record_num

training_file = '../data/dsjtzs_txfz_training.txt'
testing_file = '../data/dsjtzs_txfz_test1.txt'

tfeature = np.zeros((3000, 13)) # feature array of training data
sfeature = np.zeros(100000, 13) # feature array of testing data

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
# 3 2.x.max-x.min

tdata_delta_x = np.max(tdata_x, axis=1) - np.min(tdata_x, axis=1)
tfeature[:, 2] = tdata_delta_x.reshape((1, 3000))

sdata_delta_x = np.max(sdata_x, axis=1) - np.min(sdata_x, axis=1)
sfeature[:, 2] = sdata_delta_x.reshape((1, 3000))

# 4 3.sigma|x-x0|^2
# 5 4.|xi-xi+1|/|ti-ti+1|
# 5 5.|xi-xi+1|/|ti-ti+1|^2
# 6 6.停的次数:

tdata_x_no = np.squeeze(tdata_x, axis=2)
tdata_x_no1 = np.delete(tdata_x_no, [0], axis=1) # shape: (3000, 299)
tdata_x_non = np.delete(tdata_x_no, [299], axis=1)
tdata_t_no = np.squeeze(tdata_t, axis=2)
tdata_t_no1 = np.delete(tdata_t_no, [0], axis=1)
tdata_t_non = np.delete(tdata_t_no, [299], axis=1)
tdata_k = np.abs(np.divide(tdata_x_no1 - tdata_x_non, tdata_t_no1 - tdata_t_non))
threshold = 10e-6
record_num = count_record_num(training_or_testing="t")
for i in range(3000):
	tdata_k_tmp = tdata_k[i]
	gt_threshold = 0
	lt_threshold = 0
	for j in range(record_num[i] - 2):
		if tdata_k_tmp[j] > threshold and tdata_k_tmp[j+1] < threshold:
			++gt_threshold
		elif tdata_k_tmp[j] < threshold and tdata_k_tmp[j+1] > threshold:
			++lt_threshold
	tfeature[i, 6] = min(gt_threshold, lt_threshold)

sdata_x_no = np.squeeze(sdata_x, axis=2)
sdata_x_no1 = np.delete(sdata_x_no, [0], axis=1) # shape: (100000, 299)
sdata_x_non = np.delete(sdata_x_no, [299], axis=1)
sdata_t_no = np.squeeze(sdata_t, axis=2)
sdata_t_no1 = np.delete(sdata_t_no, [0], axis=1)
sdata_t_non = np.delete(sdata_t_no, [299], axis=1)
sdata_k = np.abs(np.divide(sdata_x_no1 - sdata_x_non, sdata_t_no1 - sdata_t_non))
threshold = 10e-6
record_num = count_record_num(training_or_testing="s")
for i in range(100000):
	sdata_k_tmp = sdata_k[i]
	gt_threshold = 0
	lt_threshold = 0
	for j in range(record_num[i] - 2):
		if sdata_k_tmp[j] > threshold and sdata_k_tmp[j+1] < threshold:
			++gt_threshold
		elif sdata_k_tmp[j] < threshold and sdata_k_tmp[j+1] > threshold:
			++lt_threshold
	sfeature[i, 6] = min(gt_threshold, lt_threshold)

# 7 7.停时有无波动:
# 8 8.折返距离:
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
sfeature[:, 9] = np.mean(ssmooth_x, axis=1).reshape((1, 3000))
sfeature[:, 10] = ssmooth_x_mse.reshape((1, 3000))

# 10 11.在x<x0时x不变的所有t的总和
# 11 12.t
