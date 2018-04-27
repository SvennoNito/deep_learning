import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 

def get_data_poly_noise(xmin, xmax, noise_rel=0.1, N=50, poly_order=1):
    
    x = (xmax - xmin) * np.random.random_sample(size=N) + xmin          # N random datapoints between ymin and ymax
    poly_coeff = 10 * np.random.random_sample(size=poly_order+1) - 5    # coefficients for the polynomial in [-5,5]
    
    #create polynomial
    y = np.zeros(x.shape)
    for i in range(poly_order+1):
        y += poly_coeff[i] * x**i
    
    noise_mag = noise_rel * np.abs((np.max(y) - np.min(y)))
    #add noise in [-noise_mag/2, noise_mag/2]
    y += noise_mag * np.random.random_sample(size=N) - noise_mag/2
    
    return (x, y)
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

xmin, xmax = -20, 30      # minimum, maximum of X
poly_order = 1            # Order of the polynomial
N_train    = 200          # Number training points

# create training data
(x_train, y_train) = get_data_poly_noise(xmin=xmin, xmax=xmax, N=N_train, noise_rel=0.2, poly_order=poly_order)
# create testing data
x_test = np.linspace(start=xmin, stop=xmax, num=1000)

# Linear regression:
################################################################################################################################
lm = linear_model.LinearRegression() # estimate m and b (that we of course would not know in real life) with a linear model from sklearn
lm.fit(x_train.reshape(-1, 1), y_train) 
print("The slopes is %s" %lm.coef_[0])
print("The intercept is %s" %lm.intercept_)
y_predict = lm.predict(x_test.reshape(-1, 1))

# why .reshape(-1, 1)?
# X needs to be transformed from [1, 2, 3, ... 299] to [[0], [1], [2], ... [299]], because:
# X -> [[feature1, feature2], [feature1, feature2], [feature1, feature2]] => here: three datapoints
# Y -> [outcome for datapoint 1, outcome for datapoint 2, for datapoint 3]

# plot training data and linear regression line (the prediction)
plt.plot(x_train, y_train, '.', label='Training Data')
plt.plot(x_test, y_predict, label='Prediction or Linear Regressoin')
plt.title('Linear Regression');
plt.legend()
plt.show()
