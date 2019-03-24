import copy
import argparse as ap
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def open_img():
    "Returns image and it's data"
    parser = ap.ArgumentParser(description="""
    Quantizes colors in image.
    Produces a modified image using these colors.""")
    
    parser.add_argument('image', action='store',
                        metavar='imagepath',
                        help='file-path of image')

    parser.add_argument('-n', '--n_colors', default=5,
                        type=int, action='store', nargs='?',
                        help="""
                        Number of colors to be quantized (default: 5)
                        (warning: choose number carefully. 
                        It may take while to run the program.)
                        """)

    args = parser.parse_args()
    img = Image.open(args.image)
    dat = np.array(img.getdata())
    n = args.n_colors
    return img, dat, n


def quantize(dat, n_colors):
    "Applies KMeans to image-data and returns most used colors"
    model = KMeans(n_clusters=n_colors).fit(dat)
    cntrs = model.cluster_centers_.astype(int)
    return cntrs


def color_diff(col1, col2):
    """Returns Euclidean distance between 2colors in RGB color space
    >>> color_diff( [200,0,0], [200,3,4])
    5.0
    """
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


def mod_img(dat, cntrs, pic):
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
    m_pic = Image.new('RGB', pic.size)
    m_pic.putdata(m_data)
    return m_pic

def main():
    picture, data, n_col = open_img()
    picture.show()
    centers = quantize(data, n_col)
    palette = palette_img(centers)
    palette.show()
    mod_picture = mod_img(data, centers, picture)
    mod_picture.show()

if __name__ == "__main__":
    main()
