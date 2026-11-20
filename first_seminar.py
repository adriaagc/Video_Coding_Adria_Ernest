import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import matplotlib.image as mpimg
from scipy.fft import fft2, ifft2, fftshift, fft, ifft, ifftshift, dct, idct
import cv2
import math

import subprocess   # to run ffmpeg commands

import argparse     # to run the .py with inputs  

FFMPEG_PATH = "C:/Users/adria/Downloads/ffmpeg-9.0.2/ffmpeg-9.0.2"

MAT8 = [[ 0,  1,  5,  6, 14, 15, 27, 28],
        [ 2,  4,  7, 13, 16, 26, 29, 42],
        [ 3,  8, 12, 17, 25, 30, 41, 43],
        [ 9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]

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


    def blur_chroma(self, yuv_img):
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

    def blur_luma(self, yuv_img):
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


    def ffmpeg_resize(self, factor, sws_falg=None):
        i = "Barns_grand_tetons"
        o = f"bicubic{factor}_{i}"
        command = [
            "ffmpeg",
            "-y",
            "-i", f"{i}.jpg",
            "-vf", f"scale=iw/{factor}:ih/{factor}", # resize the image by factor
        ]

        # If sws_flag is provided then:
        if sws_falg:
            command.extend(["-sws_flags", sws_falg])
            o = f"{sws_falg}{factor}_{i}"
            
        # Finally, append the output file
        command.append(f"{o}.jpg")

        if subprocess.run(command).returncode == 0:
            print ("FFmpeg Script Ran Successfully")
        else:
            print ("There was an error running your FFmpeg script")
        
        return

    def zigzag(self, mat8): # input a matrix of size 8x8
        # source code: https://www.geeksforgeeks.org/dsa/print-matrix-diagonal-pattern/ 
        n = len(mat8)
        if n != 8:
            print("Matrix size must be 8x8")
            return
        
        # mode - switch to derive up/down traversal
        # it - iterator count - increases until it
        # reaches n and then decreases
        mode = 0
        it = 0
        lower = 0

        ans = []

        # 2n-1 will be the number of diagonals
        for t in range(2 * n - 1):

            t1 = t

            if t1 >= n:
                mode += 1
                t1 = n - 1
                it -= 1
                lower += 1
            else:
                lower = 0
                it += 1

            for i in range(t1, lower - 1, -1):

                if (t1 + mode) % 2 == 0:
                    ans.append(mat8[i][t1 + lower - i])
                else:
                    ans.append(mat8[t1 + lower - i][i])

        print(ans)
        return ans


    def inverse_zigzag(self, num_list):
        m = len(num_list)
        if m != 64:
            print("List must contain only 64 numbers")
            return

        n = int(np.sqrt(m))
        # mode - switch to derive up/down traversal
        # it - iterator count - increases until it
        # reaches n and then decreases
        mode = 0
        it = 0
        lower = 0

        mat8 = np.zeros((n, n), dtype=int)
        idx = 0

        # a nxn matrix has 2n-1 diagonals
        for t in range(2 * n - 1):

            t1 = t

            if t1 >= n:
                mode += 1
                t1 = n - 1
                it -= 1
                lower += 1
            else:
                lower = 0
                it += 1

            for i in range(t1, lower - 1, -1):

                if (t1 + mode) % 2 == 0:
                    mat8[i][t1 + lower - i] = num_list[idx]
                    idx += 1
                else:
                    mat8[t1 + lower - i][i] = num_list[idx]
                    idx += 1

        print(mat8)
        return mat8

class DCT8x8:

     # Standard JPEG luminance quantization table
    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    def __init__(self, quality=50):
        if not 1<=quality<=100:
            raise ValueError("Quality must be between 1 and 100")
        
        self.quality = quality

        #compute the quantization scalinf factor for JPEG
        #the objective is to multiply this scaling factor to our table to obtaina
        #new quantization table depending on the quality

        if quality < 50: # if quality is lower the scale is higher and therefore we have more zeros in the resulting encoded output
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality

         # generate the scaled quantization table for our quality
        self.Q = np.floor(
            (self.QY * scale + 50) / 100
        ).astype(int)

        self.Q = np.clip(self.Q, 1, 255)

        
    
    def encode(self, block):

        block = np.asarray(block, dtype=float) #we create an array with the 64 values of our 8x8 block

        if block.shape != (8, 8): #we check if the block is 8x8
            raise ValueError("Input must be an 8x8 block")
        
        block = block - 128 #subtraact 128 to move pixel values from [0, 255] to [-128, 127] -> we center the values arround zero

        #now we apply DCT-II-> first DCT along the vertical axis (axis=0) then along the horizontal axis (axis = 1)
        vertical_dct = dct(block, axis=0, norm="ortho")
        coefficients = dct(vertical_dct, axis=1, norm= "ortho")

        #Quantize->divide each coefficient by its corresponding quantization table value and round to the nearest integer
        # small coefficients may become zero, especially at high frequencies
        quantized = np.round(coefficients / self.Q)

        return quantized.astype(int) #return the quantized 64 coeficients of the dct
    
    def decode(self, quantized):

         
        quantized = np.asarray(quantized) # we convert the input coefficients to a NumPy array.
         
        if quantized.shape != (8, 8): # we check the dimensions are 8x8
            raise ValueError("Input must be an 8x8 block")
        
        #we now apply inverse quantization -> we multiply each quantized coefficient by the corresponding value in the quantization table to recover
        #an aproximation fo the original dct coefficents
        coefficients = quantized * self.Q
        vertical_idct = dct(block, axis=0, norm="ortho")
        block = dct(vertical_idct, axis=1, norm= "ortho")

        # we undo the jpeg level shift by adding 128.
        block = block + 128

        #round the reconstructed pixel values to integers -> we clip the values from [0,255]
        return np.clip(np.round(block), 0, 255).astype(np.uint8)
    
    def run_length_encoder(self, num_list):
        encoded = []
        current_value = num_list[0]
        count= 0
        
        for value in num_list:
            value = value

            if value == current_value:
                count +=1
            else: 
                encoded.append((current_value, count))
                count= 1
                current_value = value
        
        encoded.append((current_value, count)) # we append the last value with its count

        return encoded
    
    def run_length_decoder(self, num_list):
        decoded = []    
        
        for value, count in num_list:
            decoded.extend ([value]*count) #extend adds all the elements of a list to another list one by one

        return decoded
            

if __name__ == "__main__":

    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run a specific exercise")
    parser.add_argument("exercise", type=int, choices=[2, 3, 4, 5, 6, 7, 8, 9], help="Enter an exercise number from 2 to 9")
    args = parser.parse_args()

    translator = ColorTranslator()

    if args.exercise == 2:
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


    elif args.exercise == 3:
        # Convert image to YUV space:
        original_image = plt.imread("Barns_grand_tetons.jpg") # load the RGB image; range [0-255]
        yuv_result = translator.rgb_to_yuv(original_image)

        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Show the luma blurred image:
        y_lpf_yuv = translator.blur_luma(yuv_result)
        y_lpf_rgb = translator.yuv_to_rgb(y_lpf_yuv)
        axes2[0].imshow(np.clip(y_lpf_rgb, 0, 255).astype(np.uint8))
        axes2[0].set_title("Luma Blurred")

        # 2. Show the chroma blurred image:
        Cb_Cr_lpf_yuv = translator.blur_chroma(yuv_result)
        Cb_Cr_lpf_rgb = translator.yuv_to_rgb(Cb_Cr_lpf_yuv)
        axes2[1].imshow(np.clip(Cb_Cr_lpf_rgb, 0, 255).astype(np.uint8))
        axes2[1].set_title("Chroma Blurred")

        for ax2 in axes2.flat:
            ax2.axis('off')
        
        plt.show()

    elif args.exercise == 4:
        # Default sws_falg = 'bicubic'
        translator.ffmpeg_resize(2) # resize image by 1/2
        translator.ffmpeg_resize(4) # resize image by 1/4
        translator.ffmpeg_resize(8) # resize image by 1/8

        # Compare with 'gauss':
        translator.ffmpeg_resize(2, 'gauss') # resize image by 1/2
        translator.ffmpeg_resize(8, 'gauss') # resize image by 1/8

    elif args.exercise == 5:
        # if we read that matrix with the zigzag pattern we will get
        # an ordered list of numbers from 0 to 63. 
        num_list = translator.zigzag(MAT8)
        translator.inverse_zigzag(num_list)
        
    elif args.exercise == 6:
        
        dct8x8 = DCT8x8(quality=100)
        print(dct8x8.encode(MAT8))

        dct8x8 = DCT8x8(quality=50)
        print(dct8x8.encode(MAT8))

    elif args.exercise == 7:
        block = [
            [45, -12,  8,  3,  0,  0,  0,  0],
            [-8,   6,  2,  0,  0,  0,  0,  0],
            [ 4,   2,  0,  0,  0,  0,  0,  0],
            [ 2,   0,  0,  0,  0,  0,  0,  0],
            [ 0,   0,  0,  0,  0,  0,  0,  0],
            [ 0,   0,  0,  0,  0,  0,  0,  0],
            [ 0,   0,  0,  0,  0,  0,  0,  0],
            [ 0,   0,  0,  0,  0,  0,  0,  0]
        ]
        num_list = translator.zigzag(block)
        dct8x8 = DCT8x8(quality=100)

        encoded_list= dct8x8.run_length_encoder(num_list)
        print(encoded_list)

        decoded_list = dct8x8.run_length_decoder(encoded_list)
        print(decoded_list)

        
        
