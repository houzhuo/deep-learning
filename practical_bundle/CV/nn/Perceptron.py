
# coding: utf-8

# In[2]:


import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):#  N:feature number;
        self.W = np.random.randn(N + 1)/np.sqrt(N) #+1 for bias
        self.alpha = alpha
        #divide W by the square-root of the number of inputs,
        #a common technique used to scale our weight matrix, leading to faster
        #convergence.
        
    def step(self,x):
        return 1 if x > 0 else 0
    
    def fit(self,X,y,epochs=10):
        # insert a column of 1’s as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))] #c_ mean concate ,plus one colum '1'
        
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X,y):
                p = self.step(np.dot(x, self.W))
                
                #only perform a weight update if our prediction does not right
                if p != target:
                    # determin the error
                    error = p - target
                    
                    self.W += -self.alpha * error * x  #partial(wx-y)^2
    
    def predict(self, X, addBias = True):
        #ensure input is a matrix
        X = np.atleast_2d(X)
        
        #check to see if the bias column should be added
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        
        return self.step(np.dot(X, self.W))
    

