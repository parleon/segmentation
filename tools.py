from PIL import Image
import random
import os
import numpy as np

def load_image_sample(datapath: str, count: int = None, dimensions: tuple = None, as_array: bool = True):
    imgs = os.listdir(datapath)
    
    sample_indexes = range(len(imgs))
    if count is not None: 
        sample_indexes = random.sample(sample_indexes, count)

    images = []
    for i in sample_indexes:
        img = Image.open(datapath + imgs[i])
        if dimensions is not None:
            img = img.resize(dimensions)
        if as_array:
            img = np.array(img)
        images.append(img)
    return images

def sample_center(image, sample_fraction: float = 0.25):
    center_pixel = (int(image.shape[0] // 2) , int(image.shape[1] // 2))
    pixel_offset = (int(image.shape[0] * (sample_fraction/2)), int(image.shape[1] * (sample_fraction/2)))
    return image[center_pixel[0] - pixel_offset[0]:center_pixel[0] + pixel_offset[0], center_pixel[1] - pixel_offset[1]:center_pixel[1] + pixel_offset[1]]

def flatten(image):
    if type(image) != type(np.array([])):
        image = np.array(image)
    return image.reshape(-1, 3)

def create_palette_image(cluster_centers, image_size=100):
    palette_image = Image.new("RGB", (image_size * len(cluster_centers), image_size))

    for i, center in enumerate(cluster_centers):
        color = tuple(map(int, center))
        palette_image.paste(color, (i * image_size, 0, (i + 1) * image_size, image_size))
    
    return palette_image

def classify_pixels(image_path, model, imscale=None):
    img = Image.open(image_path)
    if imscale is not None:
        img = img.resize((imscale, imscale))
    img = np.array(img)
    img_reshaped = img.reshape(-1, 3)
    predicted = model.predict(img_reshaped)
    classified_image = model.cluster_centers_[predicted].astype(int).reshape(img.shape)
    return Image.fromarray(np.uint8(classified_image))