from scipy.io import loadmat

data = loadmat("./data/denoising/dc/denoising.mat")
image = data["image"]

print(data.keys())
print(image.shape)