import matplotlib.pyplot as plt
import numpy as np

class ColorTranslator:

    def rgb_to_yuv(self, rgb_img):
        # yuv_img = rgb_img.copy()
        rgb_img = rgb_img.astype(np.float32) # more precision
        yuv_img = np.zeros_like(rgb_img)
        
        yuv_img[:,:,0] = 0.257 * rgb_img[:,:,0] + 0.504 * rgb_img[:,:,1] + 0.098 * rgb_img[:,:,2] + 16      # Y (lumma)
        yuv_img[:,:,1] = -0.148 * rgb_img[:,:,0] - 0.291 * rgb_img[:,:,1] + 0.439 * rgb_img[:,:,2] + 128    # U (Cb)
        yuv_img[:,:,2] = 0.439 * rgb_img[:,:,0] - 0.368 * rgb_img[:,:,1] - 0.071 * rgb_img[:,:,2] + 128     # V (Cr)

        return yuv_img #np.clip(yuv_img, 0, 255).astype(np.uint8)

    def yuv_to_rgb(self, yuv_img):
        rgb_img = yuv_img.astype(np.float32)

        rgb_img[:,:,0] = 1.164 * (yuv_img[:,:,0] - 16) + 1.596 * (yuv_img[:,:,2] - 128)                                     # R
        rgb_img[:,:,1] = 1.164 * (yuv_img[:,:,0] - 16) - 0.813 * (yuv_img[:,:,2] - 128) - 0.391 * (yuv_img[:,:,1] - 128)    # G
        rgb_img[:,:,2] = 1.164 * (yuv_img[:,:,0] - 16) + 2.018 * (yuv_img[:,:,1] - 128)                                     # B

        return rgb_img #np.clip(rgb_img, 0, 255).astype(np.uint8)

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