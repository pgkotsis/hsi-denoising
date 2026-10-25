import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

data = loadmat("../data/Salinas/Salinas.mat")
I1 = data["salinas"].astype(np.float64)
I1 = np.reshape(I1, (512, 217, 224), order='F')
x = 150
y = 0
I1 = I1[x:x+200, y:y+200, :]
# I1 = np.transpose(I1, (1, 2, 0)) # not to be used for salinas or indian pines images

# Subtract per-band minimum
band_min = np.min(I1, axis=(0, 1), keepdims=True)
I1 = I1 - band_min
# Divide by per-band maximum
band_max = np.max(I1, axis=(0, 1), keepdims=True)
I1 = I1 / band_max

image = I1
image_noisy = image + np.random.randn(*I1.shape) * (100 / 255)
image_noisy = np.clip(image_noisy, 0, 1)

# Save output
# savemat("./data/denoising/salinas/denoising.mat", {
#     "image": image,
#     "image_noisy": image_noisy
# })

# Visualize
image = image.transpose(2,0,1)
image_noisy = image_noisy.transpose(2,0,1)
# bands = [56, 26, 16] # DC
# bands = [169, 85, 30] # Urban (F210) - same bands as R162
# bands = [134, 80, 26] # Urban (R162) (removed bands: 1-4, 76, 87, 101-111, 136-153, 198-210)
bands = [29, 19, 9] # Salinas (removed bands: 108-112, 154-167, 224)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
ax1.imshow(np.stack((image[bands[0],:,:], image[bands[1],:,:], image[bands[2],:,:]), 0).transpose(1, 2, 0))
ax2.imshow(np.stack((image_noisy[bands[0],:,:], image_noisy[bands[1],:,:], image_noisy[bands[2],:,:]), 0).transpose(1, 2, 0))
plt.show()