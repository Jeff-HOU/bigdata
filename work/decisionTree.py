
##########################################################
 
#              Decision Tree Implementation              #

##########################################################
import numpy as np
from sklearn import tree
from datetime import date

var_save_dir = './saved_vars'
tfeature_save_file = var_save_dir + '/tfeature.npy'
sfeature_save_file = var_save_dir + '/sfeature.npy'
tlabel_save_file = var_save_dir + '/tlabel.npy'

tfeature = np.load(tfeature_save_file)
sfeature = np.load(sfeature_save_file)
tlabel = np.load(tlabel_save_file)

tlabel_squeeze = np.squeeze(tlabel, axis=1).astype(int)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(tfeature, tlabel_squeeze)
prediction = clf.predict(sfeature)

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

##                     ##
# 	Visualization Part   #
##                     ##

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, 
						out_file=dot_data,
						filled=True,
						rounded=True,
						impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("decision_tree_visualization.pdf")
