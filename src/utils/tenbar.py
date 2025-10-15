import matplotlib.pyplot as plt
import numpy as np

L = 6
k1 = 1
k2 = 0
x0 = 0

x = np.linspace(-5.99, 5.99, 1000)  
h = np.sqrt(L**2 - x**2)
P = 2 / h * ( k1*(np.arctan2(x0, h)-np.arctan2(x, h)) + k2*(np.arctan2(h, x)-np.arctan2(h, x0)) )

plt.plot(x, P)
plt.xlabel('x')
plt.ylabel('P(x)')
plt.ylim(-1, 1)
plt.title('Plot of P(x)')
plt.grid(True)
plt.show()