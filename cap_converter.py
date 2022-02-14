import glob
import numpy as np
from PIL import Image


CAP_SIZE_MM = 30
CAP_RADIUS_MM = int(CAP_SIZE_MM / 2)


def create_circular_mask(radius):
    """create a circular mask to punch out a subimage"""
    y_coord, x_coord = np.ogrid[: radius * 2, : radius * 2]
    dist_from_center = np.sqrt((x_coord - radius+0.5) ** 2 + (y_coord - radius+0.5) ** 2)
    circular_mask = dist_from_center <= radius
    return circular_mask


def convert():
    images = glob.glob("caps/*.jpg")
    for image in images:
        im = Image.open(image)
        im = im.convert('RGBA')
        im = im.resize((30, 30))
        orig_img = np.asarray(im)
        mask = create_circular_mask(CAP_RADIUS_MM)
        orig_img[:, :, 3] = mask * 255
        masked = Image.fromarray(orig_img)
        masked = masked.save(f"{image.removesuffix('.jpg')}.png")

if __name__ == "__main__":
    convert()