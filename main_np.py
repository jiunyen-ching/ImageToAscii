from PIL import Image
import os
import argparse
import numpy as np

# parameters
CHAR = np.array(list('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~i!lI;:,"^`')) # reverse this list for brightness inversion
MAX_PIXEL_VALUE = 255

def get_pixel_matrix(path):
    img = Image.open(path)
    img = img.resize((315,315))
    pixels = np.array(img.getdata())
    pixels = pixels.reshape((img.height, img.width, -1))
    return pixels[:,:,:3] # not using alpha channel, so we only return R,G,B channels

def get_brightness_matrix(pixel_matrix):
    brightness_const = np.array([0.2126, 0.7152, 0.0722])
    brightness_matrix = brightness_const * pixel_matrix # numpy takes care of broadcasting
    brightness_matrix = np.sum(brightness_matrix, axis=-1)
    return brightness_matrix

def normalize_brightness_matrix(brightness_matrix):
    max_pixel, min_pixel = np.max(brightness_matrix), np.min(brightness_matrix)
    normalized_brightness_matrix = (brightness_matrix - min_pixel) / (max_pixel - min_pixel) * MAX_PIXEL_VALUE
    return normalized_brightness_matrix

def get_ascii_matrix(brightness_matrix):
    idx = (brightness_matrix / MAX_PIXEL_VALUE * len(CHAR)).astype(np.int32) - 1
    ascii_matrix = CHAR[idx]
    return ascii_matrix

def write_ascii(ascii_matrix):
    ascii_matrix = np.repeat(ascii_matrix, repeats=2, axis=1)
    np.savetxt('output.txt', ascii_matrix, delimiter='', fmt='%s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='image file path?')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    assert os.path.exists(args.path), "I did not find the file at: " + args.path
    pixel_matrix = get_pixel_matrix(args.path)
    brightness_matrix = get_brightness_matrix(pixel_matrix)
    normalized_brightness_matrix= normalize_brightness_matrix(brightness_matrix)
    ascii_matrix = get_ascii_matrix(normalized_brightness_matrix)
    write_ascii(ascii_matrix)
    os.startfile('output.txt')

if __name__=='__main__':
    main()
