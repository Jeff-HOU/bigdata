# automatically create pdf containing seperate images and merged images.
# seperate images under ./plot_seperate_xt
# merged under ./
# shortage1: memory consuming(3.83G on my computer), but can be improved
# shortage2: parallel not supported yet
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import os
from PyPDF2 import PdfFileWriter, PdfFileReader
#from matplotlib.backends.backend_pdf import PdfPages

if not os.path.exists("plot_seperate_xt"):
	os.makedirs("plot_seperate_xt")

# Creating a routine that appends files to the output file
def append_pdf(input,output):
	[output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]

# Creating an object where pdf pages are appended to
output = PdfFileWriter()


tdata, ttarget, tlabel = fn.get_training_data()

for i in range(3000):
	tempdata = np.array([[0,0,0]])
	for j in range(300):
		if (tdata[i][j] == 0).all():
			continue
		temp = np.expand_dims(tdata[i][j], axis=0)
		tempdata = np.append(tempdata, temp, axis=0)
	tempdata = np.delete(tempdata, 0, 0)
	t_data = tempdata.transpose((1, 0))

	mpl.rcParams['legend.fontsize'] = 10
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_xlabel('x')
	ax.set_ylabel('t')
	ax.set_zlabel('y')
	ax.view_init(-90, 0)
	ax.set_title(i)
	x = t_data[0]
	y = t_data[1]
	t = t_data[2]
	x_target = np.linspace(ttarget[i][0], ttarget[i][0], 1000)
	y_target = np.linspace(np.mean(t_data[1]), np.mean(t_data[1]), 1000)
	#y_target = np.linspace(ttarget[i][1], ttarget[i][1], 1000)
	t_target = np.linspace(np.min(t_data[2]), np.max(t_data[2]), 1000)
	label = ['fake', 'real']
	plt_label = label[int(tlabel[i][0])]
	ax.plot(x, t, y, label=plt_label)
	ax.plot(x_target, t_target, y_target, label="target: "+str(ttarget[i][0])+", "+str(ttarget[i][1]))
	ax.legend()
	plt.savefig("./plot_seperate_xt/"+str(i)+".pdf", bbox_inches='tight')
	append_pdf(PdfFileReader(open("./plot_seperate_xt/"+str(i)+".pdf","rb")),output)

output.write(open("plot_xt_merged.pdf","wb"))

