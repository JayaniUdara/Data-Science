import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
	
x, t=loadData();
n,d=x.shape
w=np.random.randn(d,1)
print(x.shape, w.shape) # w is weight of each feature