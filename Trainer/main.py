import matplotlib.pyplot as plt
import numpy as np

def model(X, theta):
    return (X.dot(theta))

def cost_function(X, y, theta):
    m = len(y)
    return (1/(2*m) * np.sum((model(X, theta) - y) ** 2))

def gradient(X, y, theta):
    m = len(y)
    return (1/m * X.T.dot(model(X, theta) - y))

def gradient_descent(X, y, theta, learning_rate, n):
    cost_history = np.zeros(n)
    for i in range(n):
        theta = theta - learning_rate * gradient(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return (theta, cost_history)

def coef_deter(y, predictions):
    u = ((y - predictions)**2).sum()
    v = ((y - y.mean())**2).sum()
    return (1 - u / v)

def normalize(x):
    return ((x - min(x)) / (max(x) - min(x)))

# Reset bias values
file = open('../theta.csv', 'w+')
file.write("0.0,0.0")
file.close()

# Extract and format data from csv
data = np.genfromtxt('../data.csv', delimiter=',')[1:]
x = data[:,0]
y = data[:,1]
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

# Normalize data (turn data into a [0, 1] interval)
x = normalize(x)

# Setup X matrix
X = np.hstack((x, np.ones(x.shape)))

# Extract bias from file
theta = np.array(np.genfromtxt('../theta.csv', delimiter=','))
theta = theta.reshape(theta.shape[0], 1)

# Applied gradient descent
end_theta, cost_history = gradient_descent(X, y, theta, 0.1, 10000)

# Applied prediction to data
predictions = model(X, end_theta)

# Print data * linear line
plt.plot(x, predictions, c='r')
plt.scatter(x, y)
plt.show()

# calcul efficiency need to be near 1
print("Coef deter: " + str(coef_deter(y, predictions)))

# Save bias
file = open('../theta.csv', 'w')
file.write(str(float(end_theta[0])) + "," + str(float(end_theta[1])))
file.close()