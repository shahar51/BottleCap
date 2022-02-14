"""
Calculate how many caps are needed for a particular size
frame and stacking style
"""
import math
import glob
import configparser

import numpy as np
from PIL import Image, ImageFilter
from skimage.color import rgb2lab, lab2lch

from cap_converter import create_circular_mask


config = configparser.ConfigParser()
config.read("config.ini")

IMAGE_NAME = config["DEFAULT"]["IMAGE_NAME"]
FRAME_WIDTH_INCHES = float(config["DEFAULT"]["FRAME_WIDTH_INCHES"])
FRAME_HEIGHT_INCHES = float(config["DEFAULT"]["FRAME_HEIGHT_INCHES"])
FRAME_MARGIN_INCHES = float(config["DEFAULT"]["FRAME_MARGIN_INCHES"])
CAP_BUFFER_MM = int(config["DEFAULT"]["CAP_BUFFER_MM"])
ALIGNMENT = config["DEFAULT"]["ALIGNMENT"]
PIXEL_ART = config["DEFAULT"].getboolean("PIXEL_ART")

CAP_SIZE_MM = 30
BUFFER_DIAMETER_MM = CAP_SIZE_MM + CAP_BUFFER_MM
BUFFER_RADIUS_MM = BUFFER_DIAMETER_MM / 2
CAP_RADIUS_MM = int(CAP_SIZE_MM / 2)
CAP_WEIGHT_GRAMS = 2.2
GRAMS_TO_LBS = 2.2046e-3
INCHES_TO_MM = 25.4
FRAME_MARGIN_MM = FRAME_MARGIN_INCHES * INCHES_TO_MM
WIDTH_MM = FRAME_WIDTH_INCHES * INCHES_TO_MM - 2 * FRAME_MARGIN_MM
HEIGHT_MM = FRAME_HEIGHT_INCHES * INCHES_TO_MM - 2 * FRAME_MARGIN_MM
ASPECT_RATIO = WIDTH_MM / HEIGHT_MM

# See how the image is going to fit in the frame area we gave it
im = Image.open(IMAGE_NAME)
im_w, im_h = im.size
im_aspect_ratio = im_w / im_h
img_width_mm = min(WIDTH_MM, im_aspect_ratio * HEIGHT_MM)
img_height_mm = min(HEIGHT_MM, WIDTH_MM / im_aspect_ratio)
img_padding_x = (WIDTH_MM - img_width_mm) / 2
img_padding_y = (HEIGHT_MM - img_height_mm) / 2

# override alignment -- this needs integer scaling to work
if PIXEL_ART:
    ALIGNMENT = "grid"


def integer_spacing(values):
    """
    round the cummulative total to get evenly spaced integers
    https://stackoverflow.com/questions/13483430/how-to-make-rounded-percentages-add-up-to-100
    """
    cum_val = 0
    prev_round = 0
    need = []
    for value in values:
        cum_val += value
        cum_round = np.round(cum_val)
        need.append(int(cum_round - prev_round))
        prev_round = cum_round
    return np.array(need)


NUM_COLS = int(img_width_mm // BUFFER_DIAMETER_MM)
cap_area_width = NUM_COLS * BUFFER_DIAMETER_MM
X_PADDING_MM = (img_width_mm - cap_area_width) / 2
coords = []

adjacent_angle = 60 if ALIGNMENT == "staggered" else 0
half_chord = BUFFER_RADIUS_MM * math.sin(math.radians(adjacent_angle) / 2)
sagitta = BUFFER_RADIUS_MM - math.sqrt(BUFFER_RADIUS_MM**2 - half_chord**2)
NUM_ROWS = int((img_height_mm - 2 * sagitta) // (BUFFER_DIAMETER_MM - 2 * sagitta))
cap_stack_height = NUM_ROWS * BUFFER_DIAMETER_MM - 2 * sagitta * (NUM_ROWS - 1)
Y_PADDING_MM = (img_height_mm - cap_stack_height) / 2

match ALIGNMENT:
    case "grid":
        TOTAL_CAPS = NUM_ROWS * NUM_COLS

        # find position of each cap in pixels from top left
        for row in range(0, int(cap_stack_height), BUFFER_DIAMETER_MM):
            for col in range(0, NUM_COLS * BUFFER_DIAMETER_MM, BUFFER_DIAMETER_MM):
                coords.append((col, row))

    case "staggered":
        num_short_rows = int(NUM_ROWS // 2)
        num_long_rows = NUM_ROWS - num_short_rows
        TOTAL_CAPS = num_short_rows * (NUM_COLS - 1) + num_long_rows * NUM_COLS

        # find position of each cap in pixels from top left
        y_offset = BUFFER_DIAMETER_MM - sagitta
        if NUM_ROWS % 2 == 0:
            spacing = (cap_stack_height - 2 * y_offset) / (num_long_rows - 1)
        else:
            spacing = (cap_stack_height - BUFFER_DIAMETER_MM) / (num_long_rows - 1)
        exact_long_col = np.array(range(NUM_COLS)) * BUFFER_DIAMETER_MM
        exact_short_col = (
            np.array(range(NUM_COLS - 1)) * BUFFER_DIAMETER_MM + BUFFER_DIAMETER_MM / 2
        )
        exact_coord_long_row = np.array(range(num_long_rows)) * spacing
        exact_coord_short_row = (
            np.array(range(num_short_rows)) * spacing + y_offset - sagitta
        )

        for row in integer_spacing(exact_coord_short_row):
            for col in integer_spacing(exact_short_col):
                coords.append((col, row))
        for row in integer_spacing(exact_coord_long_row):
            for col in integer_spacing(exact_long_col):
                coords.append((col, row))

    case _:
        # option for optimal packing?
        pass

# resize image to fit physical cap area
cap_stack_height = int(math.ceil(cap_stack_height))
cap_area_width = int(cap_area_width)
im = im.resize((cap_area_width, cap_stack_height))

print(f"\nAlignment: {ALIGNMENT}")
print(f"Length: {FRAME_HEIGHT_INCHES} inches")
print(f"Width: {FRAME_WIDTH_INCHES} inches")
print(f"Columns: {NUM_COLS}")
print(f"Rows: {NUM_ROWS}")
print(f"X_padding: " f"{img_padding_x + X_PADDING_MM + FRAME_MARGIN_MM:0.3} mm")
print(f"Y_padding: " f"{img_padding_y + Y_PADDING_MM + FRAME_MARGIN_MM:0.3} mm")
print(f"Total Caps: {TOTAL_CAPS}")
print(
    f"Weight: {TOTAL_CAPS*CAP_WEIGHT_GRAMS*GRAMS_TO_LBS:0.3} lbs "
    f"({TOTAL_CAPS*CAP_WEIGHT_GRAMS/1000:0.3} kg)"
)
print(f"Actual Height: {cap_stack_height / INCHES_TO_MM:0.3} inches")
print(f"Actual Width: {cap_area_width / INCHES_TO_MM:0.3} inches")


def cap_matcher(img_array, circle_mask, cnt):
    """return the cap that best matches the circle"""
    best_score = 3 * 30 * 30 * 255
    images = glob.glob("caps/*.png")

    def score_func(Lchstd, image):
        # convert cap img to LCH space
        cap_np = np.asarray(image)
        px = circle_mask.sum() // 3
        cap_lab = rgb2lab(cap_np[:, :, :3][circle_mask == 1].reshape((1, px, 3)))
        Lchsample = lab2lch(cap_lab)

        kl, kc, kh = 1, 1, 1
        Lstd = Lchstd[:, :, 0]
        Cpstd = Lchstd[:, :, 1]
        hpstd = Lchstd[:, :, 2]

        Lsample = Lchsample[:, :, 0]
        Cpsample = Lchsample[:, :, 1]
        hpsample = Lchsample[:, :, 2]

        # Ensure hue is between 0 and 2pi
        hpstd += 2 * np.pi * (hpstd < 0)  # rollover ones that come -ve
        hpsample += 2 * np.pi * (hpsample < 0)

        # Computation of hue difference
        dhp = hpsample - hpstd
        dhp -= 2 * np.pi * (dhp > np.pi)
        dhp += 2 * np.pi * (dhp < -np.pi)

        # Compute product of chromas and locations at which it is zero for use later
        Cpprod = Cpsample * Cpstd
        zcidx = Cpprod == 0.0

        # set chroma difference to zero if the product of chromas is zero
        dhp[zcidx] = 0.0

        # weighting functions
        Lp = (Lstd + Lsample) * 0.5
        Cp = (Cpstd + Cpsample) * 0.5
        hp = (hpstd + hpsample) * 0.5

        # Identify positions for which abs hue diff exceeds 180 degrees
        hp -= (np.abs(hpstd - hpsample) > np.pi) * np.pi

        # rollover ones that come -ve
        hp += 2 * np.pi * (hp < 0)

        # Check if one of the chroma values is zero, in which case set
        # mean hue to the sum which is equivalent to other value
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]

        Lpm502 = (Lp - 50) ** 2
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = (
            1
            - 0.17 * np.cos(hp - math.radians(30))
            + 0.24 * np.cos(2 * hp)
            + 0.32 * np.cos(3 * hp + math.radians(6))
            - 0.20 * np.cos(4 * hp - math.radians(63))
        )
        Sh = 1 + 0.015 * Cp * T
        delthetarad = math.radians(30) * np.exp(-(((np.degrees(hp) - 275) / 25) ** 2))
        a = (Cp / 25) ** 7
        r_term = -np.sin(2 * delthetarad) * 2 * np.sqrt(a / (1 + a))

        L_term = (Lsample - Lstd) / (kl * Sl)
        c_term = (Cpsample - Cpstd) / (kc * Sc)
        h_term = 2 * np.sqrt(Cpprod) * np.sin(dhp * 0.5) / (kh * Sh)

        # The CIE 00 color difference
        delta_e = np.sqrt(
            L_term**2 + c_term**2 + h_term**2 + r_term * c_term * h_term
        )

        # visually we're guaranteed the best match for 50% of pixels
        # sum or mean are susceptible overall "smoothing" and outliers
        score = np.median(delta_e)
        return score, cap_np

    # convert sub img to LCH space
    img_np = np.asarray(Image.fromarray(img_array, "RGBA"))
    px = circle_mask.sum() // 3
    img_lab = rgb2lab(img_np[:, :, :3][circle_mask == 1].reshape((1, px, 3)))
    img_lch = lab2lch(img_lab)

    for cap_img in images:
        with Image.open(cap_img) as ocap:
            best_angles = []
            # find some roughly good rotations
            for angle in [0, 60, 120, 180, 240, 300]:
                score, capnp = score_func(img_lch, ocap.rotate(angle))
                # if the score is not close, don't bother rotating
                if score > best_score * 1.2:
                    break
                # no caps placed on black background
                if cap_img == "caps\\black.png":
                    if score == 0.0:
                        best_score = score
                        best_cap = capnp
                    break
                # buffer of 5% seems to give better selection..
                if score * 1.05 < best_score:
                    best_angles.append(angle)
                    best_score = score
                    best_cap = capnp
                cnt += 1

            # fine tune the rotation to 30 degree increments
            if best_angles:
                score, capnp = score_func(img_lch, ocap.rotate(best_angles[-1] + 30))
                if score < best_score:
                    best_score = score
                    best_cap = capnp
                cnt += 1
                score, capnp = score_func(img_lch, ocap.rotate(best_angles[-1] - 30))
                if score < best_score:
                    best_score = score
                    best_cap = capnp
                cnt += 1
    return best_cap, cnt


orig_img = np.asarray(im)

# blackout anything behind the transparency mask
if orig_img.shape[2] == 4:
    blackout = np.where(orig_img[:, :, 3] == 255, 1, 0)
    orig_img[:, :, 0] = orig_img[:, :, 0] * blackout
    orig_img[:, :, 1] = orig_img[:, :, 1] * blackout
    orig_img[:, :, 2] = orig_img[:, :, 2] * blackout

cnt = 0  # counting how many times score is overwritten

# create new subimage the size of a cap(uint8: 0-255)
sub_img = np.empty((CAP_RADIUS_MM * 2, CAP_RADIUS_MM * 2, 4), dtype="uint8")
mask = create_circular_mask(CAP_RADIUS_MM)
composite = Image.new("RGBA", im.size)
cap_composite = Image.new("RGBA", im.size)
for coord in coords:

    xmin, xmax = coord[0], coord[0] + CAP_RADIUS_MM * 2
    ymin, ymax = coord[1], coord[1] + CAP_RADIUS_MM * 2

    sub_img[:, :, :3] = orig_img[ymin:ymax, xmin:xmax, :3]
    sub_img[:, :, 3] = mask * 255

    # find cap image that best matches sub_img
    mask3d = np.resize(mask, ((sub_img.shape[0], sub_img.shape[1], 3)))
    cap, cnt = cap_matcher(sub_img, mask3d, cnt)

    # back to Image from numpy
    wip_img = Image.fromarray(sub_img, "RGBA")
    cap_match = Image.fromarray(cap, "RGBA")

    # paste image in position on canvas
    composite.alpha_composite(wip_img, (xmin, ymin))
    cap_composite.alpha_composite(cap_match, (xmin, ymin))
print(cnt)
composite.show()
cap_composite.show()
