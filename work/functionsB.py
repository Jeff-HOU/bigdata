import numpy as np
import pandas as pd
import os
import zipfile
from requests import get
import scipy

file_folder = "../data"
training_file = '../data/dsjtzs_txfz_training.txt'
training_file_temp = '../data/ttemp.zip'
training_file_url = "https://publicqn.saikr.com/3f96bef8dcbbf57605db3f5e79d5384e1495175270342.zip?attname=dsjtzs_txfz_training.txt.zip"
testing_file = '../data/dsjtzs_txfz_testB.txt'
testing_file_temp = "../data/stemp.zip"
testing_file_url = "https://publicqn.saikr.com/108eaf55fda29c3040328e8ef5d0b0a11499761689502.zip?attname=dsjtzs_txfz_testB.zip"

if not os.path.exists(file_folder):
	os.mkdir(file_folder)

def download_data(url, file_name):
	with open(file_name, "wb") as file:
		response = get(url)
		file.write(response.content)

def get_training_data():
	if not os.path.exists(training_file):
		download_data(training_file_url, training_file_temp)
		zf = zipfile.ZipFile(training_file_temp, "r")
		zf.extractall("/tmp/data000/")
		os.remove(training_file_temp)

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
	if not os.path.exists(testing_file):
		download_data(testing_file_url, testing_file_temp)
		zf = zipfile.ZipFile(testing_file_temp, "r")
		zf.extractall("../data/")
		os.remove(testing_file_temp)

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

		d_diff = d_max - d_min
		np.place(d_diff, d_diff == 0, 1)

		return (d - d_mean) / d_diff, param
	elif len(np.shape(d)) == 2 and np.shape(d)[1] == 1:
		return d
	else:
		param[0] = np.delete(param[0], (2), axis=1)
		param[1] = np.delete(param[1], (2), axis=1)
		param[2] = np.delete(param[2], (2), axis=1)
		param_diff = param[1] - param[2]
		np.place(param_diff, param_diff == 0, 1)
		return np.divide(d - param[0], param_diff)

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

		d_diff = d_max - d_min
		np.place(d_diff, d_diff == 0, 1)

		return (d - d_mean) / d_diff, param
	elif len(np.shape(d)) == 2 and np.shape(d)[1] == 1:
		return d
	else:
		param[0] = np.delete(param[0], (2), axis=1)
		param[1] = np.delete(param[1], (2), axis=1)
		param[2] = np.delete(param[2], (2), axis=1)
		param_diff = param[1] - param[2]
		np.place(param_diff, param_diff == 0, 1)
		return np.divide(d - param[0], param_diff)

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
'''
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	# http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
	"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	   Data by Simplified Least Squares Procedures. Analytical
	   Chemistry, 1964, 36 (8), pp 1627-1639.
	.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	   Cambridge University Press ISBN-13: 9780521880688
	"""
	#import numpy as np
	from math import factorial

	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')
'''
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	from math import factorial
	window_size = np.abs(np.int(window_size))
	order = np.abs(np.int(order))
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

def sgolay2d ( z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0
    
    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2
    
    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
    
    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
        
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band ) 
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z
    
    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band ) 
    
    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band ) 
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band ) 
    
    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')        
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')        
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

def count_record_num(training_or_testing="t"):
	if training_or_testing == "t":
		count = np.zeros((3000,))
		with open(training_file) as f:
			i = 0
			for line in f:
				parts = line.split(";")
				count[i] = len(parts) - 1
				i += 1
		f.close()
	else:
		count = np.zeros((100000,))
		with open(testing_file) as f:
			i = 0
			for line in f:
				parts = line.split(";")
				count[i] = len(parts) - 1
				i += 1
		f.close()
	return count.astype(int)
