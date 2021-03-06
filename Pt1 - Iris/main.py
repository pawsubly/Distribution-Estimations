import pandas as pd #pandas! 
import numpy as np #numpy!
import matplotlib.pyplot as plt #mpl
import seaborn as sns
from scipy.stats import multivariate_normal #used for GMM
from scipy.stats import norm #for constructing for GMM

class GMM:
    def __init__(self, k, max_iter): #initialization of GMM class 
        self.k = k 
        self.max_iter = int(max_iter)
    
    def initialvalues(self, X): #defining values before EM
        self.dimensions = X.shape 
        self.row, self.col = self.dimensions 
        self.mixingcoeff = np.full(shape=self.k, fill_value = 1/self.k) #using standard 1/k initalization
        self.weights = np.full(shape=self.dimensions, fill_value=1/self.k) #same here
        #Need to split our dataset 
        new_X = np.array_split(X, self.k)
        self.mu = [np.mean(x, axis=0) for x in new_X]
        self.sigma = [np.cov(X.T) for x in new_X]
    
    def e(self, X):
        self.weights = self.bayes(X)
        self.mixingcoeff = self.weights.mean(axis = 0)
    
    def bayes(self, X):
        likelihood = np.zeros((self.row, self.k)) #contains probabilities for every cluster in its row
        for i in range (self.k):
            distribution = multivariate_normal(
                mean = self.mu[i],
                cov = self.sigma[i],
                allow_singular=True) # actually, this should be removed but I can't figure out the bug
            likelihood[:,i] = distribution.pdf(X)

        numerator = likelihood * self.mixingcoeff
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def m(self, X):
        for i in range (self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, aweights = (weight/total_weight).flatten(), bias = True)

    def fit(self , X):
        self.initialvalues(X)

        for iteration in range(self.max_iter):
            self.e(X)
            self.m(X)
    
    def predict(self, X):
        weights = self.bayes(X)
        return np.argmax(weights, axis = 1) 

df = pd.read_excel('iris.xls')

sepal_column = df.loc[:,'Petal Length (cm)']
sepalwidth_column = df.loc[:, 'Petal Width (cm)']
nums = sepal_column.values
nums2 = sepalwidth_column.values
X = np.vstack((nums,nums2))

gmm = GMM(k=2, max_iter = 10)
gmm.fit(X)
labels = gmm.predict(X)

'''
sns.scatterplot(x=nums, y=nums2, hue=df.loc[:, 'Class'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title("GMM Potential")
plt.show()
'''

y = np.array([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]) #sepal length bins

s = np.random.normal(np.mean(nums2), np.std(nums2, ddof=1), 1000)
# plt.scatter(s, 1/(stdev * np.sqrt(2 * np.pi)) * np.exp( - (s - mean)**2 / (2 * stdev**2) ), color='r')
plt.hist(x=df['Petal Length (cm)'], bins = 10, density=True, histtype='bar') #plotting our Sepal Length data
plt.xlabel('Petal Length (cm)')
plt.ylabel('Probability Density')
plt.title('Petal Length')

plt.show()