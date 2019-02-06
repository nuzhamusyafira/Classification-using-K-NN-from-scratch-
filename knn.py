import math
import numpy as np
csv2 = np.genfromtxt('pima.csv', usecols=(-1), delimiter=",")
csv = np.genfromtxt('pima.csv', usecols=range(8), delimiter=",")
n=len(csv)
print("Amount of datas:",n)
k=input("Enter k: ")
k=int(k)
part=int(n/10)
part2=0
if n%10!=0:
	part2=n%10
result=[]
from scipy.spatial import distance
for x in range(10-part2):
	print("Loading iteration...", x+1)
	for y in range(x*part,part*(x+1)):
		arr=[]
		for z in range(x*part):
			a=csv[y]
			b=csv[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		for z in range(x*part,part*(x+1)):
			inf=math.inf
			arr.append(inf)
		for z in range(part*(x+1),n):
			a=csv[y]
			b=csv[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		knn=np.array(arr)
		knn=np.argsort(knn)[:k]
		zero=0
		one=0
		for i in range(k):
			if csv2[knn[i]]==0:
				zero+=1
			else:
				one+=1
		if one==zero:
			result.append(csv2[knn[0]])
		elif one>zero:
			result.append(1)
		else:
			result.append(0)
for x in range(10-part2,10):
	print("Loading iteration...", x+1)
	for y in range(x*part,part*(x+1)+1):
		arr=[]
		for z in range(x*part):
			a=csv[y]
			b=csv[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		for z in range(x*part,part*(x+1)+1):
			inf=math.inf
			arr.append(inf)
		for z in range(part*(x+1)+1,n):
			a=csv[y]
			b=csv[z]
			dst=distance.cosine(a,b)
			arr.append(dst)
		knn=np.array(arr)
		knn=np.argsort(knn)[:k]
		zero=0
		one=0
		for i in range(k):
			if csv2[knn[i]]==0:
				zero+=1
			else:
				one+=1
		if one==zero:
			result.append(csv2[knn[0]])
		elif one>zero:
			result.append(1)
		else:
			result.append(0)
print("Iterations completed")
counter=0
for x in range(n):
	if result[x]==csv2[x]:
		counter+=1
acc=100*counter/n
print("Correctly Classified Instances:",counter)
print("Incorrectly Classified Instances:",n-counter)
print("Accuracy:",acc,"%")
