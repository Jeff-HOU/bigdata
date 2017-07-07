import numpy as np
import os
from functions import get_training_and_testing_data, scale_data, scale_one_data, \
					  count_record_num

# training_file = '../data/dsjtzs_txfz_training.txt'
# testing_file = '../data/dsjtzs_txfz_test1.txt'

var_save_dir = './saved_vars'
tfeature_save_file = var_save_dir + '/tfeature'
sfeature_save_file = var_save_dir + '/sfeature'
tlabel_save_file = var_save_dir + '/tlabel'

tfeature = np.zeros((3000, 4)) # feature array of training data
sfeature = np.zeros((100000, 4)) # feature array of testing data

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
# 1 0.

# 2 1. num of absolute delta_x = 0, the back part may contain noise
utdata_x_diff=np.diff(utdata_x[:,:,0])
record_num_t = count_record_num(training_or_testing="t")
for i in range(3000):
    num_of_zero_dx = 0
    for j in range(record_num_t[i] - 1):
        if utdata_x_diff[j] == 0:
            num_of_zero_dx+=1
    tfeature[i, 1] = num_of_zero_dx

usdata_x_diff=np.diff(usdata_x[:,:,0])
record_num_s = count_record_num(training_or_testing="s")
for i in range(100000):
    num_of_zero_dx = 0
    for j in range(record_num_s[i] - 1):
        if usdata_x_diff[j] == 0:
            num_of_zero_dx+=1
    sfeature[i, 1] = num_of_zero_dx

#print(tfeature[i,1])
#end of idea 2

# 3 2. linear regression, calculate residuals (sum of squared Euclidean 2-norm)
tmp_t = np.concatenate((tdata_t, np.ones_like(tdata_t)), axis=-1)
for i in range(tdata_t.shape[0]):
	tfeature[i, 2] = np.linalg.lstsq(tmp_t[i], tdata_x[i])[1]
tmp_s = np.concatenate((sdata_t, np.ones_like(sdata_t)), axis=-1)
for i in range(sdata_t.shape[0]):
	sfeature[i, 2] = np.linalg.lstsq(tmp_s[i], sdata_x[i])[1]

''' For Visualization, please use the following code instead (set i before you plot):
a = np.concatenate((tdata_t, np.ones_like(tdata_t)), axis=-1)
b = np.zeros((3000,))
c = np.zeros((3000,2,1))
for i in range(tdata_t.shape[0]):
	b[i] = np.linalg.lstsq(a[i], tdata_x[i])[1]
	c[i] = np.linalg.lstsq(a[i], tdata_x[i])[0]
i = 0
import matplotlib.pyplot as plt
plt.plot(tdata_t[i], tdata_x[i], 'o', label='Original data', markersize=10)
plt.plot(tdata_t[i], c[i,0]*tdata_t[i]+c[i,1], 'r', label='Fitted line')
plt.legend()
plt.show()
'''
# end of idea 3

# 4 3. stop times

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
	tfeature[i, 3] = min(gt_threshold, lt_threshold)

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
	sfeature[i, 3] = min(gt_threshold, lt_threshold)

# end of idea 4

if not os.path.exists(var_save_dir):
	os.mkdir(var_save_dir)
np.save(tfeature_save_file, tfeature)
np.save(sfeature_save_file, sfeature)
np.save(tlabel_save_file, tlabel)

