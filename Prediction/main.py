import matplotlib.pyplot as plt
import numpy as np

def normalize(x, init):
    return ((x - min(init)) / (max(init) - min(init)))

# Extract bias
try:
    theta = np.genfromtxt('../theta.csv', delimiter=',')
except OSError:
    print("Run train program first")
    exit(1)

# Extract data from csv
data = np.genfromtxt('../data.csv', delimiter=',')[1:]

# Get input value
kms = input("Enter kilometers: ")
try:
    kms = int(kms)
except:
    print("\nValue need to be numeric !")
    exit(1)

# Get prediction
price = theta[1] + theta[0] * normalize(kms, data[:,0])
print("Estimate price: " + str(int(price)))

# Get linear values
x = np.array(range(int(max(data[:,0]))))
y = theta[1] + theta[0] * normalize(x, data[:,0])

################
# Format chart #
################

# Print data
plt.scatter(data[:,0], data[:,1], c='blue')

# Print calculated value
plt.scatter(kms, price, c='red')

# Print line
plt.plot(x, y, c='red')

# Print legend
plt.title("Cars")
plt.xlabel("Kilometers")
plt.ylabel("Price")
plt.show()
