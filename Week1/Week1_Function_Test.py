import numpy as np
import matplotlib.pyplot as plt

def sample_linear(num_samples=5):
    x_eval = np.linspace(-3, 3, 100)

    for i in range(num_samples):
        w = np.random.normal(0, 1, 1)
        f = w*x_eval
        plt.plot(x_eval, f)
    plt.savefig('linear.png')
    plt.close()

def sample_neuralnet(num_samples=5):
    x_eval = np.linspace(-3, 3, 100).reshape((1,100))

    for i in range(num_samples):
        w1 = np.random.normal(0, 1, (10, 1))
        w2 = np.random.normal(0, 1, (1, 10))

        activation = lambda x : 1/(1+np.exp(-x))

        f = w2 @ activation(w1 @ x_eval)

        plt.plot(x_eval[0,:], f[0,:])
    plt.savefig('neuralnet.png')
    plt.close()


sample_linear()
sample_neuralnet()