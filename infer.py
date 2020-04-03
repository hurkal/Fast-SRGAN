from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
import cv2
import os
import shutil

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')
parser.add_argument('--opencv_resize', type=str, help='True, to create an addition copy with OpenCV resize()')


def main():
    args = parser.parse_args()

    # Get all image paths
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    # Change model input shape to accept all size inputs
    model = keras.models.load_model('models/generator.h5')
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    # Loop over all images
    for image_path in image_paths:
        
        # Read image
        low_res = cv2.imread(image_path, 1)
        print(image_path)

        # Convert to RGB (opencv uses BGR as default)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Rescale to 0-1.
        low_res = low_res / 255.0

        # Get super resolution image"
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = ((sr + 1) / 2.) * 255

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        # Save the results:
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)

        if (args.opencv_resize=='True'): 
            not_gan_filename = args.output_dir+"/not_gan_"+image_path.split("/")[1]
            not_gan_image = cv2.imread(image_path,1)
            print(not_gan_filename + " - " + str(not_gan_image.shape[0])+"x"+str(not_gan_image.shape[1]))
    
            width = int(not_gan_image.shape[0]*4)
            height = int(not_gan_image.shape[1]*4)
            dim = (width, height)
            not_gan_image = cv2.resize(not_gan_image,dim)
            cv2.imwrite(not_gan_filename,not_gan_image)
        
        
        


if __name__ == '__main__':
    main()
