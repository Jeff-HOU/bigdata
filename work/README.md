# bdc_2017
This is a competition project for [2017中国高校计算机大赛––––大数据挑战赛(Big Data Challenge)](http://bdc.saikr.com/vse/bdc/2017)<br>
The task is to differentiate between a computer behaviour and a human behaviour based on a series of mouse trace data which contains:<br>

* mouse movement trace - x, y, t
* target point - x, y

## Packages used:
* numpy
* pandas
* sklearn.tree
* sklearn.externals.six.StringIO
* pydotplus
* datetime
* os
* zipfile
* requests
* scipy
* matplotlib
* mpl_toolkits.mplot3d.Axes3D
* PyPDF2
* joblib
* multiprocessing
Last two are used to speed up the visualization of original data. A simple tutorial can be found [here](https://blog.dominodatalab.com/simple-parallelization/)

## special technique: [savitzky_golay](http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html)