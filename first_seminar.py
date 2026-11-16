import matplotlib.pyplot as plt
import numpy as np

class ColorTranslator:

    def rgb_to_yuv(self, rgb_img):
        # yuv_img = rgb_img.copy()
        rgb_img = rgb_img.astype(np.float32)
        yuv_img = np.zeros_like(rgb_img)
        
        yuv_img[:,:,0] = 0.257 * rgb_img[:,:,0] + 0.504 * rgb_img[:,:,1] + 0.098 * rgb_img[:,:,2] + 16 # Y (lumma)
        yuv_img[:,:,1] = -0.148 * rgb_img[:,:,0] - 0.291 * rgb_img[:,:,1] + 0.439 * rgb_img[:,:,2] + 128
        yuv_img[:,:,2] = 0.439 * rgb_img[:,:,0] - 0.368 * rgb_img[:,:,1] - 0.071 * rgb_img[:,:,2] + 128

        return np.clip(yuv_img, 0, 255).astype(np.uint8)

    def yuv_to_rgb(self, yuv_img):
        rgb_img = yuv_img.copy()

        rgb_img[:,:,0] = 1.164 * (yuv_img[:,:,0] - 16) + 1.596 * (yuv_img[:,:,2] - 128) 
        rgb_img[:,:,1] = 1.164 * (yuv_img[:,:,0] - 16) - 0.813 * (yuv_img[:,:,2] - 128) - 0.391 * (yuv_img[:,:,1] - 128)
        rgb_img[:,:,2] = 1.164 * (yuv_img[:,:,0] - 16) + 1.596 * (yuv_img[:,:,2] - 128) + 2.018 * (yuv_img[:,:,1] - 128)

        return np.clip(rgb_img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # 1. Instantiate the class
    translator = ColorTranslator()

    # 2. Load an image (make sure you have an image file named 'photo.jpg' in your directory)
    # Matplotlib loads JPEGs as RGB NumPy arrays
    original_image = plt.imread("Barns_grand_tetons.jpg")

    # 3. Process the image
    yuv_result = translator.rgb_to_yuv(original_image)
    rgb_restored = translator.yuv_to_rgb(yuv_result)

    # 4. Visualize the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original RGB")
    
    axes[1].imshow(yuv_result)
    axes[1].set_title("YUV Space (Visualized as RGB)")
    
    axes[2].imshow(rgb_restored)
    axes[2].set_title("Restored RGB")
    
    for ax in axes:
        ax.axis('off')
        
    plt.show()