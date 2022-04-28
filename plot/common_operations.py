import os, sys
sys.path.append(os.getcwd())

import numpy as np
import plot.plotting_params as params
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import img_as_ubyte
import imageio


def savefig(plt, path, show_message= True, tight_flag= True, pad_inches= 0, newline= True):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches= pad_inches)
    else:
        plt.savefig(path)
    if newline:
        print("")

def open_gif_writer(file_path, duration= 0.5):
    print("=> Saving to {}".format(file_path))
    gif_writer = imageio.get_writer(file_path, mode='I', duration= duration)

    return gif_writer

def convert_fig_to_ubyte_image(fig):
    canvas = FigureCanvas(fig)
    # draw canvas as image
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    image = img_as_ubyte(image)

    return image

def add_ubyte_image_to_gif_writer(gif_writer, ubyte_image):

    gif_writer.append_data(ubyte_image)

def close_gif_writer(gif_writer):

    gif_writer.close()
