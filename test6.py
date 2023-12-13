import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#load dataset 
# read data
#df = pd.read_csv('data.csv')

# display the first 10 elements of the dataframe
#print(df.head(10))


#X= df['Hours']
#X=np.array(df['Hours'])
#n=X.shape[0]
#X=X.reshape(n,1)
#t= np.array(df['Scores']).reshape(n,1)
#print(X)
#print(t)
#print(n)

def loadData():
	df = pd.read_csv('data.csv')
	X=np.array(df['Hours'])
	n=X.shape[0]
	X=X.reshape(n,1)
	t= np.array(df['Scores']).reshape(n,1)
	onesv=np.ones((n,1))
	#to avoid bias term seperate as parameter we need to append ones
	X=np.append(X,onesv, axis=1)
	return X, t
def visualize(x,t, yp=None):
		plt.scatter(x,t)
		plt.title("Data")
		plt.xlabel('Hours')
		plt.ylabel('Scores')
		if yp is not None:
			plt.scatter(x, yp, c='red')
		plt.show()
		
X, t =loadData();
print(X.shape)
#visualize(X[:,:-1],t)

def plotlosses(losses):
	fig=plt.figure(figsize=(8,6))
	X=list(range(len(losses))) 
	plt.plot(X, losses,'r-')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()
		
n,d=X.shape
#w=np.random.randn(n,1)
w=np.random.randn(d,1)
print(X.shape,w.shape)

#lr=0.001
lr=0.01
nepoch=30
losses=[]

for i in range(nepoch):
	#calculate prediction
	y=np.dot(X,w) #y=Xw
	
	#calculate derivative
	dldw=np.dot(X.T, y-t)/n
	#print(dldw)
	
	#update w
	w=w- lr*dldw
	
	#calculate loss
	loss=np.mean((y-t)**2)
	#print(loss)
	
	losses.append(loss)
	visualize(X[:,:-1],t,y)
	exit

y=np.dot(X, w)
plotlosses(losses);
#visualize(X[:,:-1],t,y)
#print(w)

