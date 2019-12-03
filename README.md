# INF_552 (Machine Learning of Data Science) Homeworks Respority

#### Practical applications of machine learning techniques to real-world problems. Uses in data mining and recommendation systems and for building adaptive user interfaces.

---
## Homework_1 (KNN)
This Biomedical data set was built by Dr. Henrique da Mota during a medical residence
period in Lyon, France. Each patient in the data set is represented in the data set
by six biomechanical attributes derived from the shape and orientation of the pelvis
and lumbar spine (in this order): pelvic incidence, pelvic tilt, lumbar lordosis angle,
sacral slope, pelvic radius and grade of spondylolisthesis. The following convention is
used for the class labels: DH (Disk Hernia), Spondylolisthesis (SL), Normal (NO) and
Abnormal (AB). In this exercise, we only focus on a binary classification task NO=0
and AB=1

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_1/pdf/Homework1-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_1/Aaron_homewrok_1.ipynb" target="_blank">[Notebook Ipynb]</a>

## Homework_2 (Linear Regression)
The dataset contains data points collected from a Combined Cycle Power Plant over
6 years (2006-2011), when the power plant was set to work with full load. Features
consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP),
Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical
energy output (EP) of the plant.

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_2/pdf/Homework2-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_2/Aaron_homework_2.ipynb" target="_blank">[Notebook Ipynb]</a>

## Homework_3 (Logistic Regression)
An interesting task in machine learning is classification of time series. In this problem,
we will classify the activities of humans based on time series obtained by a Wireless
Sensor Network.

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_3/pdf/Homework3-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_3/Aaron_homework_3.ipynb" target="_blank">[Notebook Ipynb]</a>


## Homework_4 (Regularization | Forest)
- 1. The LASSO and Boosting for Regression
	Download the Communities and Crime data1 from https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime. 
	Use the first 1495 rows of data as the training set and the rest as the test set.

- 2. Tree-Based Methods
	Download the APS Failure data from: https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks. The dataset contains a training set and a test set. The training set contains 60,000 rows, of which 1,000 belong to the positive class and 171 columns, of which one is the class column. All attributes are numeric.

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_4/pdf/Homework4-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_4/Aaron_homework_4.ipynb" target="_blank">[Notebook Ipynb]</a>

## Homework_5 (SVM | k-means)
- 1. Multi-class and Multi-Label Classification Using Support Vector Machines
	Download the Anuran Calls (MFCCs) Data Set from: https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29. 
	Choose 70% of the data randomly as the training set

- 2. K-Means Clustering on a Multi-Class and Multi-Label Data Set
	Monte-Carlo Simulation: Perform the following procedures 50 times, and report the average and standard deviation of the 50 Hamming Distances that you calculate.

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_5/pdf/Homework5-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_5/Aaron_homework_5.ipynb" target="_blank">[Notebook Ipynb]</a>

## Homework_6 (Passive Learning | Active Learning)
- 1. Supervised, Semi-Supervised, and Unsupervised Learning
	Download the Breast Cancer Wisconsin (Diagnostic) Data Set from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29. Download the data in https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data, which has IDs, classes (Benign=B, Malignant=M), and 30 attributes. This data has two output classes. Use the first 20% of the positive and negative classes in the file as the test set and the rest as the training set.

- 2. Active Learning Using Support Vector Machines
	Download the banknote authentication Data Set from: https://archive.ics.uci.edu/ml/datasets/banknote+authentication. Choose 472 data points randomly as the test set, and the remaining 900 points as the training set. This is a binary classification problem.

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_6/pdf/Homework6-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_6/Aaron_homework_6.ipynb" target="_blank">[Notebook Ipynb]</a>

## Homework_7 (Deep Learning LSTM | CNN)
- 1. Generative Models for Text
	In this problem, we are trying to build a generative model to mimic the writing style of prominent British Mathematician Philosopher, prolific writer, and political activist, Bertrand Russell.

- 2. (Deep) CNNs for Image Colorization
	This assignment uses a convolutional neural network for image colorization which turns a grayscale image to a colored image.5 By converting an image to grayscale, we loose color information, so converting a grayscale image back to a colored version is not an easy job. We will use the CIFAR-10 dataset. Downolad the dataset from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_7/pdf/Homework7-inf552.pdf" target="_blank">[Homework pdf]</a> | 
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_7/Generative_Models_For_Text.ipynb" target="_blank">[LSTM Ipynb]</a>
<a href="https://github.com/AaronYang2333/INF_552/blob/master/ay_hw_7/CNNs_For_Image_Coloization.ipynb" target="_blank">[CNNs Ipynb]</a>