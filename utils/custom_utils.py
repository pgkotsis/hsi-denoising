import numpy as np
import pywt
from .common_utils import *

def get_noisy_image(img_np, sigma, laplace=0):
    '''Adds Gaussian noise to an image.
    
    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    '''
    
    if sigma == 0 and laplace == 0:
        return img_np.copy().astype(np.float32)

    if laplace > 0:
        img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape) 
                               + np.random.laplace(scale=laplace, size=img_np.shape), 0, 1).astype(np.float32)
    else:
        img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    return img_noisy_np

def create_sparse_noise(img, k, return_indices=False):
    '''Replace k% of values in the input NumPy array with 1.0 at random positions.'''

    if k == 0:
        if return_indices:
            return img.copy(), []
        else:
            return img.copy()

    img_copy = img.copy()
    total_elements = img_copy.size
    noisy_elements = int(total_elements * k / 100)
    indices = np.random.choice(total_elements, noisy_elements, replace=False)

    flat_img = img_copy.flatten()
    flat_img[indices] = 1.0
    img_noised_sparse = flat_img.reshape(img_copy.shape)

    if return_indices:
        indices = np.unravel_index(indices, img_copy.shape)
        index_list = list(zip(*indices))
        return img_noised_sparse, index_list
    else:
        return img_noised_sparse

def _add_stripe_noise(img, rng):
    indices = rng.integers(10, img.shape[1] - 10, 10)
    value = rng.uniform(0.5, 1)
    img[:, indices] = value
    return img

def create_stripe_noise(img, k, seed=42):
    """Creates stripe noise in k out of all the image bands."""
    rng = np.random.default_rng(seed)

    if k == 0:
        return img.copy(), []

    indices = rng.integers(0, img.shape[0], k)
    img_noised_stripe = img.copy()

    for index in indices:
        _add_stripe_noise(img_noised_stripe[index, :, :], rng)

    return img_noised_stripe, indices

# def _add_stripe_noise(img):

#     indices = np.random.randint(10, img.shape[1] - 10, 10)
#     value = np.random.uniform(0.5, 1)
#     img[:, indices] = value

#     return img

# def create_stripe_noise(img, k):
#     '''Creates stripe noise in k out of all the image bands.'''

#     if k == 0:
#         return img.copy(), []

#     indices = np.random.randint(0, img.shape[0], k)
#     img_noised_stripe = img.copy()

#     for index in indices:
#         _add_stripe_noise(img_noised_stripe[index, :, :])
    
#     return img_noised_stripe, indices

def sigma_from_snr(SNR, image, option=False):
    '''Calculates standard deviation from the SNR value.'''

    if SNR == np.inf:
        return 0

    power = np.mean(image ** 2)
    opt = 2 if option else 1

    return np.sqrt(power * ((10) ** (-SNR / (10 * opt))))

def mpsnr(image_gt, image_pred):
    '''Calculates MPSNR given a target image and a predicted image.'''

    assert image_gt.shape == image_pred.shape

    mse = np.mean((image_gt - image_pred) ** 2, axis = (1, 2))
    mxm = np.max(image_gt ** 2, axis = (1, 2))
    
    if np.any(np.isclose(mse, 0.0)):
        return np.inf
    else:
        return np.mean(10 * np.log10(mxm / mse))

def nmse(image_gt, image_pred):
    '''Calculates NMSE given a target image and a predicted image.'''

    assert image_gt.shape == image_pred.shape

    se = np.sum((image_gt - image_pred) ** 2, axis=(1, 2))
    n = np.sum(image_gt ** 2, axis=(1, 2))

    return np.mean(se / n)

def tv_loss(x):
    '''Calculates TV loss for an image `x`.'''

    dh = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    dw = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

    tv = torch.sum(dh) + torch.sum(dw)

    return tv / x.numel()

def sstv_loss(x):
    '''Calculates SSTV loss for an image `x`.

    See, e.g.:
        https://ieeexplore.ieee.org/document/7397913
        https://ieeexplore.ieee.org/document/9534671
    '''

    ds = x[:,1:,:,:] - x[:,:-1,:,:]
    dh = torch.abs(ds[:,:,:,1:] - ds[:,:,:,:-1])
    dw = torch.abs(ds[:,:,1:,:] - ds[:,:,:-1,:])
    
    sstv = torch.sum(dh) + torch.sum(dw)

    return sstv / x.numel()

def sigma_estimate(img_noisy_np):
    '''Estimate noise std for a hyperspectral image using MAD in the HH subband.'''

    B, H, W = img_noisy_np.shape
    sigma_per_band = []

    for b in range(B):

        band = img_noisy_np[b, :, :]
        coeffs = pywt.dwt2(band, wavelet='db1')
        LL, (LH, HL, HH) = coeffs

        sigma_b = np.median(np.abs(HH)) / 0.6745
        sigma_per_band.append(sigma_b)

    return np.mean(sigma_per_band)

def sure_loss(img_pred, img_noisy, sigma, model, epsilon=1e-3, norm=2):
    '''Computes the Monte Carlo SURE loss for a denoising neural network.'''
    
    N = img_noisy.numel()
    if norm == 2:
        residual = torch.sum((img_noisy - img_pred) ** 2)
    elif norm == 1:
        residual = torch.sum(torch.abs(img_noisy - img_pred))
    elif norm == 0.5:
        residual = l_half_norm(img_pred, img_noisy)
    else:
        raise ValueError("Unsupported norm value. Use 2, 1, or 0.5.")

    b = torch.randn_like(img_noisy)
    img_perturbed = img_noisy + epsilon * b
    img_pred_perturbed = model(img_perturbed)
    divergence = torch.sum((img_pred_perturbed - img_pred) * b) / epsilon

    sure = (residual - N * sigma**2 + 2 * sigma**2 * divergence) / N

    return sure

def l_half_norm(img_pred, img_noisy):
    '''Computes the L_{1/2} norm of the difference between two tensors.'''

    diff = torch.abs(img_pred - img_noisy)
    loss = torch.sum(torch.pow(diff, 0.5)) / diff.numel()

    return loss