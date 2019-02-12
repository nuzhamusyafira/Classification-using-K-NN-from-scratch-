import math
import numpy as np
# we extract the last column as it appears to be the class and name it as y_train,
# y_train indicates what our predicted class will be compared to
y_train = np.genfromtxt('pima.csv', usecols=(-1), delimiter=",")

# then we define the rest, which are predictors, as x_train
x_train = np.genfromtxt('pima.csv', usecols=range(8), delimiter=",")

# we find out the number of records and save it to n
n=len(x_train)
print("Amount of datas:",n)

# we retrieve the k that we want within input
k=input("Enter k: ")
k=int(k)

# because we're using 10-folds cross validation to generate a more distributed train,
# we first determine the number of records each fold has
fold_num=int(n/10)

# if our data isn't divisible by 10, then we put the rest to rest_data
# example, if we have 500 records (divisible by 10), then:
# 50 50 50 50 50 50 50 50 50 50
# but if we have 524, then:
# 53 53 53 53 52 52 52 52 52 52.
rest_data=0
if n%10!=0:
	rest_data=n%10

# y_test will later hold the result of our predictions to be compared to y_train
y_test=[]

# for the distance measurements, we'll use distance form scipy.spatial
from scipy.spatial import distance

# rest_data contains the value of modulus 10, then in the first n-rest_data,
# we're gonna add an addition iteration of n/10 (fold_num)
for x in range(10-rest_data):
	print("Loading iteration...", x+1)
	for y in range(x*fold_num,fold_num*(x+1)):
		arr=[]
		for z in range(x*fold_num):
			a=x_train[y]
			b=x_train[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		for z in range(x*fold_num,fold_num*(x+1)):
			inf=math.inf
			arr.append(inf)
		for z in range(fold_num*(x+1),n):
			a=x_train[y]
			b=x_train[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		knn=np.array(arr)
		knn=np.argsort(knn)[:k]
		zero=0
		one=0
		for i in range(k):
			if y_train[knn[i]]==0:
				zero+=1
			else:
				one+=1
		if one==zero:
			y_test.append(y_train[knn[0]])
		elif one>zero:
			y_test.append(1)
		else:
			y_test.append(0)
for x in range(10-rest_data,10):
	print("Loading iteration...", x+1)
	for y in range(x*fold_num,fold_num*(x+1)+1):
		arr=[]
		for z in range(x*fold_num):
			a=x_train[y]
			b=x_train[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		for z in range(x*fold_num,fold_num*(x+1)+1):
			inf=math.inf
			arr.append(inf)
		for z in range(fold_num*(x+1)+1,n):
			a=x_train[y]
			b=x_train[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		knn=np.array(arr)
		knn=np.argsort(knn)[:k]
		zero=0
		one=0
		for i in range(k):
			if y_train[knn[i]]==0:
				zero+=1
			else:
				one+=1
		if one==zero:
			y_test.append(y_train[knn[0]])
		elif one>zero:
			y_test.append(1)
		else:
			y_test.append(0)
print("Iterations completed")
counter=0
for x in range(n):
	if y_test[x]==y_train[x]:
		counter+=1
acc=100*counter/n
print("Correctly Classified Instances:",counter)
print("Incorrectly Classified Instances:",n-counter)
print("Accuracy:",acc,"%")
