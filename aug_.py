import pandas as pd
import numpy as np
import os
import cv2

import random
import Augmentor
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil

columns = ['x','y']
j = 0
x_data = []
x_1_data = []
y_data = []
aug = []
img_set = []
augmented_images = []
augmented_image = []
augmented_point = []
augmented_points = []
rotation = 0

grid_width = 4
grid_height = 4
magnitude = abs(8)

dx = random.randint(-magnitude, magnitude)
dy = random.randint(-magnitude, magnitude)
def perform_operation(images, i_, k_ ):
    """
    Distorts the passed image(s) according to the parameters supplied during
    instantiation, returning the newly distorted image.

    :param images: The image(s) to be distorted.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
        PIL.Image.
    """
    image = np.array(images)

    images = []
    for i in range(len(image)):
        img = image[i].reshape((550,550))
        images.append(Image.fromarray(img.astype('uint8'),'L'))
    w, h = images[0].size

    horizontal_tiles = grid_width
    vertical_tiles = grid_height

    width_of_square = int(floor(w / float(horizontal_tiles)))
    height_of_square = int(floor(h / float(vertical_tiles)))

    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])

    # For loop that generates polygons could be rewritten, but maybe harder to read?
    # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

    # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

    last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    def do(image):

        for a, b, c, d in polygon_indices:


            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                            x2, y2,
                            x3 + dx, y3 + dy,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                            x2 + dx, y2 + dy,
                            x3, y3,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                            x2, y2,
                            x3, y3,
                            x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                            x2, y2,
                            x3, y3,
                            x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []

    for image in images:
        augmented_images.append(do(image))
    
    for l in range(len(df_img)):
        img_ = np.array(augmented_images[l])
        img_ = img_.reshape(-1)
        augmented_point.append(np.argmax(img_))
    for  l in range(len(augmented_point)):
        x = augmented_point[l]%550
        y = int(augmented_point[l]/550)
        augmented_points.append([x,y])
    dataframe = pd.DataFrame(augmented_points, columns= ['x','y'])
    
    dataframe['label'] = i_
    dataframe.to_csv("./DATA/aug/{}augs_{}".format(i_,k_), index=False)
    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []
    img_set = []        

    return augmented_images

for i in range(10):
  
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
    
  rotation = random.randint(-20,20)

  while(rotation == 0):
    rotation = random.randint(-20,20)

  for k in range(10):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((550, 550, 1), np.uint8)
    img_ = np.zeros((550, 550, 1), np.uint8)

    for l in range(len(df_img)):
      img_[df_img[l][1]][df_img[l][0]] = 255
      img_set.append(img_)
      img_ = np.zeros((550, 550, 1), np.uint8)

    perform_operation(img_set,i,k)





