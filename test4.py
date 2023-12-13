import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
#df=pd.read_csv('data.csv'); #read data
#(1)x=df['Hours']; #x is our design matrix n is no of samples
#t=df['Scores'];
#print(x);
#print(t);
def visualize(x,t):
	plt.scatter(x,t)
	plt.title('Data')
	plt.xlabel('Hours')
	plt.ylabel('Scores')
	plt.show()

def loadData():
	df=pd.read_csv('data.csv');
	x=np.array(df['Hours'])
	n=x.shape[0]
	x=x.reshape(n,1)
	t=np.array(df['Scores']).reshape(n,1)
	onesv=np.ones((n,1))
	x=np.append(x, onesv, axis=1)
	return x,t

def plotLosses(losses):
	fig=plt.figure(figsize=(8,6))
	x=list(range(len(losses)))
	plt.plot(x, losses, 'r-')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()
	
x, t=loadData();
#print(x.shape)
#visualize(x[:,:-1], t) #: consider all rows,1- leaving last column
n,d=x.shape
w=np.random.randn(d,1)
#print(x.shape, w.shape) # w is weight of each feature

#we have to iterate
lr=0.001
nepoch=30

for i in range(nepoch):
	y=np.dot(x,w) #y=Xw first calculate y
	dldw=np.dot(x.T, y-t)/n	# then calculate derivative
	print(dldw)
	w=w - lr*dldw
	loss=np.mean((y-t)**2)
	losses.append(loss)
	print(losses)