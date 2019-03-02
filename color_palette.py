import sys
import copy
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def open_img():
    "Returns image and it's data"
    if len(sys.argv) == 2:
        img = Image.open(sys.argv[1])
    else:
        img = Image.open("example.jpg")
    dat = np.array(img.getdata())
    return img, dat


def quantize(dat, n_colors):
    "Applies KMeans to image-data and returns most used colors"
    model = KMeans(n_clusters=n_colors).fit(dat)
    cntrs = model.cluster_centers_.astype(int)
    return cntrs


def color_diff(col1, col2):
    "Returns Euclidean distance between 2colors in RGB color space"
    return ((col1[0] - col2[0])**2 +
            (col1[1] - col2[1])**2 +
            (col1[2] - col2[2])**2)**0.5


def palette_img(cntrs):
    "Returns image of color palette"
    width, height = 128, 128
    palette = Image.new('RGB', (width*len(cntrs), height))
    for i, _ in enumerate(cntrs):
        col = Image.new("RGB", (width, height), tuple(cntrs[i]))
        palette.paste(col, (width*i, 0))
    return palette


def mod_img(dat, cntrs):
    "Returns image in range of color palette"
    modified_data = copy.deepcopy(dat)
    for pix, _ in enumerate(dat):
        col_dic = {}
        for col in cntrs:
            col_dic[tuple(col)] = color_diff(dat[pix], col)
        # Finds min of col_dic and replaces
        for key, value in col_dic.items():
            if min(col_dic.values()) == value:
                modified_data[pix] = key
    # Creates image
    m_data = list(tuple(pixel) for pixel in modified_data)
    m_pic = Image.new('RGB', picture.size)
    m_pic.putdata(m_data)
    return m_pic


if __name__ == "__main__":
    picture, data = open_img()
    picture.show()
    centers = quantize(data, 5)
    palette = palette_img(centers)
    palette.show()
    mod_picture = mod_img(data, centers)
    mod_picture.show()
