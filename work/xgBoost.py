import numpy as np
from xgboost import XGBClassifier
from datetime import date
from sklearn.grid_search import GridSearchCV
var_save_dir = './saved_vars'
tfeature_save_file = var_save_dir + '/tfeature.npy'
sfeature_save_file = var_save_dir + '/sfeature.npy'
tlabel_save_file = var_save_dir + '/tlabel.npy'

tfeature = np.load(tfeature_save_file)
sfeature = np.load(sfeature_save_file)
tlabel = np.load(tlabel_save_file)

model = XGBClassifier(min_child_weight=12, subsample=0.5, colsample_bytree=0.8, eta=0.1)
model.fit(tfeature, tlabel)
prediction = model.predict(sfeature).astype(int)

num_of_black_samples = 100000 - np.count_nonzero(prediction)
black_samples = np.zeros(num_of_black_samples)
j = 0
k = 1
for i in range(100000):
	if prediction[i] == 0:
		black_samples[j] = k
		j += 1
	k += 1
d = date.today().timetuple()
fname = '../submit/BDC0539_' + str(d[0]).zfill(4) + str(d[1]).zfill(2) + str(d[2]).zfill(2) + '.txt'
np.savetxt(fname, black_samples, fmt='%d', delimiter='\n')
