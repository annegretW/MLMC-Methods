import matplotlib.pyplot as plt
import numpy as np

e = [1e-2,8e-3,6e-3,4e-3,2e-3]

samples_1 = [1914, 67, 38]
samples_2 = [2964, 79, 51]
samples_3 = [5537, 123, 80, 50]
samples_4 = [15955, 405, 310, 237, 225, 104]
samples_5 = [58436, 1565, 925, 610, 576, 250]


samples = [samples_1,samples_2,samples_3,samples_4,samples_5]
costs_mlmc = np.zeros(5)
    
for i in range(len(samples)):
    c = 0
    for j in range(len(samples[i])):
        c += 64*(2**i)*samples[i][j]
    costs_mlmc[i] = c

print(costs_mlmc)

plt.plot(e,costs_mlmc)
plt.xlabel(r'Genauigkeit $\epsilon$')
plt.ylabel(r'$\epsilon^2$-Kosten')
plt.show()

