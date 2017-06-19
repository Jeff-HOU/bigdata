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