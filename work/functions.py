import numpy as np
import pandas as pd

training_file = '../data/dsjtzs_txfz_training.txt'
testing_file = '../data/dsjtzs_txfz_test.txt'

def get_training_data():
	df = pd.read_csv(training_file, sep=' ', header=None)
	df.columns = ['id', 'data', 'target', 'label']
	
	n = len(df)
	data = np.zeros((n, 300, 3), dtype=np.int32)
	target = np.zeros((n, 2), dtype=np.float32)
	label = np.zeros((n, 1))
	label[:, 0] = df['label']
	for i, p in enumerate(df['data']):
		for j, q in enumerate(p.split(';')[:-1]):
			data[i][j] = list(map(int, q.split(',')))
	for i, p in enumerate(df['target']):
		target[i] = list(map(float, p.split(',')))
	return data, target, label

def get_testing_data():
	df = pd.read_csv(testing_file, sep=' ', header=None)
	df.columns = ['id', 'data', 'target']
	
	n = len(df)
	data = np.zeros((n, 300, 3), dtype=np.int32)
	target = np.zeros((n, 2), dtype=np.float32)
	for i, p in enumerate(df['data']):
		for j, q in enumerate(p.split(';')[:-1]):
			data[i][j] = list(map(int, q.split(',')))
	for i, p in enumerate(df['target']):
		target[i] = list(map(float, p.split(',')))
	return data, target

def scale_one_data(d, param=[]):
	if len(np.shape(d)) == 3:
		d_transpose = d.transpose((0, 2, 1))

		d_mean = d_transpose.mean(axis=2)
		param.append(d_mean)
		d_mean = np.expand_dims(d_mean, axis=1)
		d_mean = np.repeat(d_mean, [300], axis=1)

		d_max = d_transpose.max(axis=2)
		param.append(d_max)
		d_max = np.expand_dims(d_max, axis=1)
		d_max = np.repeat(d_max, [300], axis=1)

		d_min = d_transpose.min(axis=2)
		param.append(d_min)
		d_min = np.expand_dims(d_min, axis=1)
		d_min = np.repeat(d_min, [300], axis=1)

		return (d - d_mean) / (d_max - d_min), param
	elif len(np.shape(d)) == 2 and np.shape(d)[1] == 1:
		return d
	else:
		param[0] = np.delete(param[0], (2), axis=1)
		param[1] = np.delete(param[1], (2), axis=1)
		param[2] = np.delete(param[2], (2), axis=1)
		return (d - param[0]) / (param[1] - param[2])

def scale_one_data_1(d, param=[]):
	if len(np.shape(d)) == 3:
		d_transpose = d.transpose((0, 2, 1))

		d_mean = d_transpose.mean(axis=2)
		param.append(d_mean)
		d_mean = np.expand_dims(d_mean, axis=1)
		d_mean = np.repeat(d_mean, [300], axis=1)

		d_max = d_transpose.max(axis=2)
		param.append(d_max)
		d_max = np.expand_dims(d_max, axis=1)
		d_max = np.repeat(d_max, [300], axis=1)

		d_min = d_transpose.min(axis=2)
		param.append(d_min)
		d_min = np.expand_dims(d_min, axis=1)
		d_min = np.repeat(d_min, [300], axis=1)

		return (d - d_mean) / (d_max - d_min), param
	elif len(np.shape(d)) == 2 and np.shape(d)[1] == 1:
		return d
	else:
		param[0] = np.delete(param[0], (2), axis=1)
		param[1] = np.delete(param[1], (2), axis=1)
		param[2] = np.delete(param[2], (2), axis=1)
		return (d - param[0]) / (param[1] - param[2])

def scale_data(tdata, ttarget, tlabel, sdata, starget):
	tdata_scaled, param_1 = scale_one_data(tdata)
	ttarget_scaled = scale_one_data(ttarget, param_1)
	tlabel_scaled = scale_one_data(tlabel)
	sdata_scaled, param_2 = scale_one_data_1(sdata)
	starget_scaled = scale_one_data_1(starget, param_2)
	return tdata_scaled, \
		   ttarget_scaled, \
		   tlabel_scaled, \
		   sdata_scaled, \
		   starget_scaled

def get_training_and_testing_data():
	tdata, ttarget, tlabel = get_training_data()
	sdata, starget = get_testing_data()
	return tdata, ttarget, tlabel, sdata, starget

def get_scaled_training_and_testing_data():
	tdata, ttarget, tlabel = get_training_data()
	sdata, starget = get_testing_data()
	return scale_data(tdata, ttarget, tlabel, sdata, starget)
