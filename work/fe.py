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

tfeature = np.zeros((3000, 4)) # feature array of training data
sfeature = np.zeros((100000, 4)) # feature array of testing data

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
# 1 0.


if not os.path.exists(var_save_dir):
	os.mkdir(var_save_dir)

np.save(tfeature_save_file, tfeature)
np.save(sfeature_save_file, sfeature)
np.save(tlabel_save_file, tlabel)

