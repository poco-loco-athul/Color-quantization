import sys
import copy
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# open image

# if len(sys.argv) == 2:
#     im = Image.open(sys.argv[1])
# else:    
#     # opens image
#     im = Image.open("example.jpg")
im = Image.open("example.jpg")


# get image data
data = np.array(im.getdata())
modified_data = copy.deepcopy(data)


# Kmeans learning
kmeans = KMeans(n_clusters=5, max_iter=1).fit(data)
centers = kmeans.cluster_centers_.astype(int)

def color_difference(col1,col2):
    "Returns Euclidean distance between 2colors in RGB color space"
    return ((col1[0] - col2[0])**2 +
            (col1[1] - col2[1])**2 +
            (col1[2] - col2[2])**2)**0.5


# Image of color palette
width, height = 128, 128
palette = Image.new('RGB', (width*len(centers), height))
for i in range(len(centers)):
    col = Image.new("RGB", (width, height), tuple(centers[i]))
    palette.paste(col, (width*i,0))


# Makes image-data in range of color palette
for pix in range(len(modified_data)):
    col_dic = {}
    for col in centers:
        col_dic[tuple(col)] = color_difference(modified_data[pix],col)
    for key, value in col_dic.items():
        if min(col_dic.values()) == value:
            modified_data[pix] = key

modified_im = Image.new('RGB', im.size)
m_data = list(tuple(pixel) for pixel in modified_data)
modified_im.putdata(m_data)


if __name__ == "__main__":
    im.show()
    palette.show()
    modified_im.show()    
