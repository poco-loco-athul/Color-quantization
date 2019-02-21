from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# opens image
im = Image.open("example.jpg")

# get pixel values from image
data = np.array(im.getdata())

# Kmeans learning
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
centers = kmeans.cluster_centers_.astype(int)


# Showing output

a = tuple(centers[0])
out = Image.new("RGB", (128, 128), a)
out.show()




