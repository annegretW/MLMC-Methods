import matplotlib.pyplot as plt
import numpy as np

e = [3e-2,1e-2,8e-3,7e-3,6e-3,5e-3,2e-3,4e-3]
x = [0,1,2,3,4,5,6]
samples_1 = [122, 20, 20]
samples_3 = [1914, 67, 38]
samples_4 = [2964, 79, 51]
samples_5 = [4018, 86, 33, 24]
samples_6 = [5537, 123, 80, 50]
samples_7 = [10355, 236, 146, 206, 112, 50]
samples_8 = [58436, 1565, 925, 610, 576, 250]
samples_9 = [15955, 405, 310, 237, 225, 104]

plt.yscale('log')
#plt.plot(x[:len(samples_1)],samples_1)
plt.plot(x[:len(samples_3)],samples_3)
plt.plot(x[:len(samples_4)],samples_4)
#plt.plot(x[:len(samples_5)],samples_5)
plt.plot(x[:len(samples_6)],samples_6)
plt.plot(x[:len(samples_9)],samples_9)
plt.plot(x[:len(samples_8)],samples_8)
#plt.legend([r'$\epsilon = 0.03$','$\epsilon = 0.01$','$\epsilon = 0.008$','$\epsilon = 0.007$','$\epsilon = 0.006$','$\epsilon = 0.005$'])
plt.legend([r'$\epsilon = 0.01$','$\epsilon = 0.008$','$\epsilon = 0.006$','$\epsilon = 0.004$','$\epsilon = 0.002$'])
plt.xlabel('Level l')
plt.ylabel(r'$N_l$')
plt.show()
