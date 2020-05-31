from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt

from surfaces import u3


def surface2im(u, v):
    """calculates images of given surface
    u - surface (heights matrix)
    v - light vector
    """
    kernel_x = torch.tensor(
        [[[0, 0, 0], [1, 0, -1], [0, 0, 0]]]
    )
    kernel_y = torch.tensor(
        [[[0, 1, 0], [0, 0, 0], [0, -1, 0]]]
    )
    def kernelize(kernel):
        kernel = torch.reshape(kernel, (1, 1, 3, 3))
        kernel = kernel.double()
        return kernel
    kernel_x = kernelize(kernel_x)
    kernel_y = kernelize(kernel_y)

    surf_x = torch.nn.functional.conv2d(u, kernel_x)
    surf_y = torch.nn.functional.conv2d(u, kernel_y)
    img = (
        (v[0] * surf_x + v[1] * surf_y - v[2])
        / (torch.sqrt(1 + surf_x**2 + surf_y**2) * (v[0]**2 * v[1]**2 * v[2]**2)**.5)
    )
    return img

def score(guess, e, v, per_pixel=False):
    """calculates cost function of given surface
    guess - surface to measure
    e - image of original surface
    v - light vector used to iluminate original surface
    per_pixel - if True, cost value is divided by number comparision points
    """
    e_guess = surface2im(guess, v)
    cost = (e_guess - e)**2
    if per_pixel:
        cost = torch.average(cost)
    return torch.sum(cost)

n = 100
l_rate = 0.1
surf = torch.tensor(u3(n))
surf = torch.reshape(surf, (1, 1, n, n))

v = [1, 1, 1]

img = surface2im(surf, v)

noisy_surf2 = torch.rand(surf.shape) * 1.5 + surf

# initial guess
guess = torch.tensor(noisy_surf2, requires_grad=True)

for i in range(2000):
    energy = score(guess, img, v)
    if (i+1) % 1000 == 0:
        print(energy)
    energy.backward()

    guess.data -= guess.grad.data * l_rate
    guess.grad.zero_()


print(surf.min(), surf.max())
print(guess.min(), guess.max())
print(noisy_surf2.min(), noisy_surf2.max())
guess.requires_grad_(False)
fig, axs = plt.subplots(1,3)
# # print(torch.max(surf), torch.min(surf))
# print(type(img), img)
# axs[0,0].imshow(torch.reshape(surf_x, [*surf_x.shape[2:5]]))
# axs[0,1].imshow(torch.reshape(surf_y, [*surf_y.shape[2:5]]))
axs[1].imshow(torch.reshape(guess, [*guess.shape[2:4]]), vmin=-.5, vmax=2.5)
axs[0].imshow(torch.reshape(noisy_surf2, [*noisy_surf2.shape[2:4]]), vmin=-.5, vmax=2.5)
axs[2].imshow(torch.reshape(surf, [*surf.shape[2:4]]), vmin=-.5, vmax=2.5)
# axs[1,0].imshow(torch.reshape(surf, (n, n)))
# axs[1,1].imshow(torch.reshape(img, [*img.shape[2:5]]))
plt.show()

