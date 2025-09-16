import numpy as np
import matplotlib.pyplot as plt

"""
functions used: 

np.linspace(start=a, stop=b, num=n)
    Creates an array of values in range [a, b] of size n, where each value is evenly spaced out.
    Space length = (b - 1)/(n - 1)

np.random.normal(loc, scale, size) :
    Draw a random sample from a normal (gaussian) distribution
    loc = mean
    scale = standard deviation
    size = return vector size. If no size given, draw one sample, return as a scalar.
"""

def sample_linear(num_samples=5):
    x_eval = np.linspace(-3, 3, 100) # Linspace, 1 x 100

    for i in range(num_samples):
        w = np.random.normal(0, 1, 1) # draw from N.D, m = 0, sd = 1. Return a 1.D vector
        f = w*x_eval # Broadcasting, non-standard matrix multiplication
        plt.plot(x_eval, f) # Plot
    plt.savefig('linear.png')
    plt.close()

def sample_neuralnet(num_samples=5):
    x_eval = np.linspace(-3, 3, 100).reshape((1,100)) # 1 x 100

    for i in range(num_samples):
        w1 = np.random.normal(0, 1, (10, 1)) # Draw 10 from N.D, m = 0, sd = 1, return 10 x 1 Vector
        w2 = np.random.normal(0, 1, (1, 10)) # Draw 10 from N.D, m = 0, sd = 1, return 1 x 10 Vector

        a_f = lambda x : 1/(1+np.exp(-x)) # define the ACTIVATION function: Sigmoid

        f = w2 @ a_f(w1 @ x_eval)
        # Breakdown
        # w1 x x_eval
        # This time, w1 is multiplied by x_eval using actual matrix multiplication
        # w1 is a 10 x 1 matrix, and x_eval, this time is 1 x 100. This results in a 10 x 100 matrix.
        # By passing the resulting matrix through the activation function, the function is applied element wise.
        # w2 is then multiplied by the resulting matrix, leaving a 1 x 100 matrix.

        plt.plot(x_eval[0,:], f[0,:])
        # x_eval[0, :] -> Take the 0th row, include ALL columns.
        # f[0,:] -> Take the 0th row, include ALL columns.
        # Note that both x_eval and f are of size 1 x 100! Reason why we have numpy slicing is because of nested lists
        # (x_eval = [[...]], f = [[...]], so we want to get rid of one set of brackets to get JUST an array)
    plt.savefig('neuralnet.png')
    plt.close()

sample_linear()
sample_neuralnet()




