# bdc_2017
This is a competition project for [2017中国高校计算机大赛––––大数据挑战赛(Big Data Challenge)](http://bdc.saikr.com/vse/bdc/2017)<br>
The task is to differentiate between a computer behaviour and a human behaviour based on a series of mouse trace data which contains:<br>

* mouse movement trace - x, y, t
* target point - x, y
## Project details
We extract 16 features and use Decision Tree Classifier to deal with testing data.<br>

**Extracted features:**
1. (x.max-x.min)/(t.max-t.min)
2. Distance between the x-axis of the last point in each record and that of the target point
3. x.max-x.min
4. sigma|x - x0|^2
5. how many same time-point that a data possess
6. the mean of velocity
7. the standard deviation of velocity
8. mean(|xi-xi+1|/|ti-ti+1|^2) (mean of acceleration)
9. standard deviation of accelration
10. 停的次数
11. 停时有无波动
12. 折返距离
13. 光滑度
14. 光滑度方差
15. 在x<x0时x不变的所有t的总和
16. judging the similarity of the line with a straight line there are two cases, one with a direct line, another with x remaining the same first and then followed by a direct sloping line. Then it is natural to focus on the regression of the direct line only.

## Packages used
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

Last two packages are used to speed up the visualization of original data. A simple tutorial can be found [here](https://blog.dominodatalab.com/simple-parallelization/)

## Special technique
* [savitzky_golay](http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html) to smooth a graph.
