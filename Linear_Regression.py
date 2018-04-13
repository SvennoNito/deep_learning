import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 

# create random dataset with linear relationship between x and y
xmax    = 50
x, m, b = np.random.random(300)*xmax, 2., -7                          # random x-vector from 0 to xmax 
y       = [np.random.normal(0,5)+2*x_+b for x_ in  x]                 # m*x + b + gaussian_noise

# estimate m and b (that we of course would not know in real life) with a linear model from sklearn
lm = linear_model.LinearRegression()
# X needs to be transformed from [1, 2, 3, ... 299] to [[0], [1], [2], ... [299]], because:
# X -> [[feature1, feature2], [feature1, feature2], [feature1, feature2]] => here: three datapoints
# Y -> [outcome for datapoint 1, outcome for datapoint 2, for datapoint 3]
lm.fit(x.reshape(-1, 1), y)                  
print("The slopes is %s" %lm.coef_[0])
print("The intercept is %s" %lm.intercept_)

# predict y from regression line
x_ = np.random.random(300)*xmax
y_ = lm.predict(x_.reshape(-1, 1))

# plot training data and linear regression line (the prediction)
plt.plot(x, y, 'x',label='Training Data')
plt.plot(x_, y_, label='Prediction or Linear Regressoin')
plt.title('Linear Regression'); plt.xlabel('X'); plt.ylabel('Y'); plt.legend()
plt.show()
 
