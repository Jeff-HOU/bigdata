# label by hand
# if mse > 1 human, else machine
import numpy as np
from datetime import date
var_save_dir = './saved_vars'
tfeature_save_file = var_save_dir + '/tfeature.npy'
sfeature_save_file = var_save_dir + '/sfeature.npy'
tlabel_save_file = var_save_dir + '/tlabel.npy'

tfeature = np.load(tfeature_save_file)
sfeature = np.load(sfeature_save_file)
tlabel = np.load(tlabel_save_file)
tlabel_squeeze = np.squeeze(tlabel, axis=1).astype(int)
tmp=np.concatenate((np.expand_dims(np.array(range(100000)),axis=-1), sfeature),axis=-1)
tmp1=(tmp[tmp[:,1]<1][:,0]+1).astype(int)
d = date.today().timetuple()
fname = '../submit/BDC_' + str(d[0]).zfill(4) + str(d[1]).zfill(2) + str(d[2]).zfill(2) + '.txt'
np.savetxt(fname, tmp1, fmt='%d', delimiter='\n')