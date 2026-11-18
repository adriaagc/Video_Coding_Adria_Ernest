import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import matplotlib.image as mpimg
from scipy.fft import fft2, ifft2, fftshift, fft, ifft, ifftshift
import cv2
import math

class ColorTranslator:

    def rgb_to_yuv(self, rgb_img):
        # yuv_img = rgb_img.copy()
        rgb_img = rgb_img.astype(np.float32) # more precision
        yuv_img = np.zeros_like(rgb_img)
        
        yuv_img[:,:,0] = 0.257 * rgb_img[:,:,0] + 0.504 * rgb_img[:,:,1] + 0.098 * rgb_img[:,:,2] + 16      # Y (luma)
        yuv_img[:,:,1] = -0.148 * rgb_img[:,:,0] - 0.291 * rgb_img[:,:,1] + 0.439 * rgb_img[:,:,2] + 128    # U (Cb)
        yuv_img[:,:,2] = 0.439 * rgb_img[:,:,0] - 0.368 * rgb_img[:,:,1] - 0.071 * rgb_img[:,:,2] + 128     # V (Cr)

        return yuv_img #np.clip(yuv_img, 0, 255).astype(np.uint8)

    def yuv_to_rgb(self, yuv_img):
        rgb_img = yuv_img.astype(np.float32)

        rgb_img[:,:,0] = 1.164 * (yuv_img[:,:,0] - 16) + 1.596 * (yuv_img[:,:,2] - 128)                                     # R
        rgb_img[:,:,1] = 1.164 * (yuv_img[:,:,0] - 16) - 0.813 * (yuv_img[:,:,2] - 128) - 0.391 * (yuv_img[:,:,1] - 128)    # G
        rgb_img[:,:,2] = 1.164 * (yuv_img[:,:,0] - 16) + 2.018 * (yuv_img[:,:,1] - 128)                                     # B

        return rgb_img #np.clip(rgb_img, 0, 255).astype(np.uint8)


def blur_chroma(yuv_img):
# 1. apply a 2D FFT to Cb and Cr channels
    Cb = yuv_img[:,:,1]
    fft_Cb = fft2(Cb)
    fft_shifted_Cb = fftshift(fft_Cb) # low freq at the center  
    Cr = yuv_img[:,:,2]
    fft_Cr = fft2(Cr)
    fft_shifted_Cr = fftshift(fft_Cr) # low freq at the center
# 2. create the LPF: a black image with a white circle on the center
    center = (yuv_img.shape[1]//2, yuv_img.shape[0]//2)
    radius = 34     # this cotrols the cutoff frequency
    color = 1       # white
    thickness = -1  # paint the inside of the circle 
    im = np.zeros((yuv_img.shape[0], yuv_img.shape[1]), dtype=np.uint8)
    lpf = cv2.circle(im, center, radius, color, thickness)
# 3. apply the LPF to both images:
    Cb_LPF = np.real(ifft2(ifftshift(fft_shifted_Cb * lpf)))

    Cr_LPF = np.real(ifft2(ifftshift(fft_shifted_Cr * lpf)))

# 4. create the blurred image:
    blur_img = yuv_img.copy() # the Y (luma) channel will stay untouched
    blur_img[:,:,1] = Cb_LPF
    blur_img[:,:,2] = Cr_LPF

    return blur_img

def blur_luma(yuv_img):
# 1. apply a 2D FFT to Y channel
    Y_channel = yuv_img[:,:,0]
    fft_Y = fft2(Y_channel)
    fft_shifted_Y = fftshift(fft_Y) # low freq at the center  
# 2. create the LPF: a black image with a white circle on the center
    center = (yuv_img.shape[1]//2, yuv_img.shape[0]//2) # (horizontal , vertical) coordiantes
    radius = 34     # this cotrols the cutoff frequency
    color = 1       # white
    thickness = -1  # paint the inside of the circle 
    im = np.zeros((yuv_img.shape[0], yuv_img.shape[1]), dtype=np.uint8)
    lpf = cv2.circle(im, center, radius, color, thickness)
# 3. apply the LPF to both images:
    Y_lpf = np.real(ifft2(ifftshift(fft_shifted_Y * lpf)))

# 4. create the blurred image:
    blur_img = yuv_img.copy() # the Cb and Cr channels will stay untouched
    blur_img[:,:,0] = Y_lpf

    return blur_img

if __name__ == "__main__":

    translator = ColorTranslator()

    original_image = plt.imread("Barns_grand_tetons.jpg") # load the RGB image; range [0-255]

    # Convert image to YUV space and recover it to RGB again:
    yuv_result = translator.rgb_to_yuv(original_image)
    rgb_restored = translator.yuv_to_rgb(yuv_result)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].imshow(original_image)
    axes[0,0].set_title("Original RGB")
    
    axes[0,1].imshow(np.clip(yuv_result, 0, 255).astype(np.uint8))
    axes[0,1].set_title("YUV Space (Visualized as RGB)")
    
    axes[1,0].imshow(np.clip(rgb_restored, 0, 255).astype(np.uint8))
    axes[1,0].set_title("Restored RGB")

    # visualize the quantization error (when clamping and casting to an integer):
    abs_error = np.abs(rgb_restored - original_image.astype(np.float32))
    mae = np.mean(abs_error, axis=-1) # mean absolute error of each pixel (mean over the last dimension)
    min_err = np.min(mae)
    max_err = np.max(mae)
    im = axes[1,1].imshow(mae/max_err, cmap='inferno')
    axes[1,1].set_title("Mean Absolute Error: |recovered - original|")
    fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

    print(f"Minimum MAE: {min_err}")
    print(f"Maximum MAE: {max_err}")
    
    for ax in axes.flat:
        ax.axis('off')
        
    plt.show()


    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Show the luma blurred image:
    y_lpf_yuv = blur_luma(yuv_result)
    y_lpf_rgb = translator.yuv_to_rgb(y_lpf_yuv)
    axes2[0].imshow(np.clip(y_lpf_rgb, 0, 255).astype(np.uint8))
    axes2[0].set_title("Luma Blurred")

    # 2. Show the chroma blurred image:
    Cb_Cr_lpf_yuv = blur_chroma(yuv_result)
    Cb_Cr_lpf_rgb = translator.yuv_to_rgb(Cb_Cr_lpf_yuv)
    axes2[1].imshow(np.clip(Cb_Cr_lpf_rgb, 0, 255).astype(np.uint8))
    axes2[1].set_title("Chroma Blurred")

    for ax2 in axes2.flat:
        ax2.axis('off')
        
    plt.show()