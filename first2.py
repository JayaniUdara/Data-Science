import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

def plotData(x,y,y_hat=None):
	plt.scatter(x,y,marker='.')
	if y_hat is not None:
		plt.scatter(x,y_hat,marker='v')
	plt.title('Data')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
	
def getData(n):
	x=np.random.randn(n,1)
	y=2*x + 1 + np.random.randn(n,1)
	return x,y
	
n = 100
x_ori, y = getData(n)
X = np.append(x_ori, np.ones((n,1)), axis=1)
w =  np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))#closed form solution
y_hat=np.dot(X,w)#make predictions
plotData(x_ori,y,y_hat)