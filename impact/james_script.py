import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 

np.random.seed(1010)

def model(x, p):
	return p[0]*np.exp(p[1]*x)+p[2]

xdata   = np.linspace(-4, 4, 2**16)
y       = model(xdata, [5, 1, 4.0])
y_noise = 10 * np.random.normal(size=xdata.size)
ydata   = y + y_noise



def obj(p):
	ymodel = model(xdata, p)
	RMSE   = np.sqrt(np.mean((ymodel - ydata)**2))
	return RMSE

x0  = [2,0,1]  #IC
res = minimize(obj, x0, method='Nelder-Mead', tol=1e-6)

#PLOT THE RESULT 
yfit = model(xdata, res.x)
plt.plot(xdata, ydata, 'o', xdata, yfit, '-')
plt.show()