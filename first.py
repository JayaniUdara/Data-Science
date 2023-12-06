import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

def plotData(x,y):
	plt.scatter(x,y)
	plt.title('Data')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
	
def getData(n):
	x=np.random.randn(n,1)
	y=2*x + 1 + np.random.randn(n,1)
	return x,y
	
#x=np.random.randn(n,1)
#ones=np.ones((n,1))
#k=np.append(x,ones,axis=0)



#plotData(x,y)

#design matrix
n = 100
x_ori, y = getData(n)
X = np.append(x_ori, np.ones((n,1)), axis=1)
w =  np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
print(w)