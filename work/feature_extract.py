import numpy as np
from functions import get_scaled_training_and_testing_data, \
					  get_training_and_testing_data, scale_data, scale_one_data, \
					  savitzky_golay

tfeature_file = "../data/training_data_feature.csv"
sfeature_file = "../data/testining_data_feature.csv"

tfeature = np.zeros((3000, 13))
#sfeature = np.zeros(100000, 11)

# 0.(normalization)
utdata, uttarget, utlabel, usdata, ustarget = get_training_and_testing_data() # get unscaled training and testing data
utdata_x = np.delete(utdata, [1, 2], axis=2)								  # track's x axis of unscaled training data
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
t_data_delta_x = np.max(tdata_x, axis=1) - np.min(tdata_x, axis=1)
tfeature[:, 2] = t_data_delta_x.reshape((1, 3000))
# 4 3.sigma|x-x0|^2
# 5 4.|xi-xi+1|/|ti-ti+1|
# 5 5.|xi-xi+1|/|ti-ti+1|^2
# 6 6.停的次数:
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
smooth_x = np.abs(tdata_s[:, :, 0] - tdata[:, :, 0])
smooth_x_mse = (smooth_x ** 2).mean(axis=1) # SOME NAN AFTER THIS STEP!!!!
											# SEE LATER IF NEEDS FIXED
											# np.count_nonzero(~np.isnan((smooth_x_mse)) --> NON-NAN: 2924
tfeature[:, 10] = smooth_x_mse.reshape((1, 3000))

# 10 11.在x<x0时x不变的所有t的总和
# 11 12.t
