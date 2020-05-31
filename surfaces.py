import numpy as np 
import matplotlib.pyplot as plt

def u3(n):
    res = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            x = i/(n-1)
            y = j/(n-1)
            res[i, j] = 1 - np.tanh((4-6*x+3*x**2-6*y+3*y**2)*25/6)
    
    return res


if __name__ == "__main__":
    data = u3(12)
    print(data)
    plt.imshow(data)
    plt.show()
    # sns.lmplot(data=u3(5))