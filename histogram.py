import numpy as np
import PIL
from matplotlib import pyplot
def augment(img):
    img = 255 - img
    #img = img / np.max(img)
    #img = np.power(img, 3)
    #img = img * 255.
    return img

def show_histogram(img_path):
    img = PIL.Image.open(img_path)
    img = img.convert("L")
    img = np.array(img)
    img = augment(img)
    img = img.astype(np.uint8)
    histogram = np.zeros([256], dtype=np.float32)
    print(img.shape)
    for pixel in img:
        histogram[pixel] += 1
    pyplot.figure()
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(img, cmap='gray')
    pyplot.title("Original image")
    pyplot.subplot(1, 2, 2)
    pyplot.bar(np.arange(len(histogram)), histogram)
    pyplot.title("histogram")
    pyplot.show()

if __name__ == "__main__":
    show_histogram("capture.jpg")