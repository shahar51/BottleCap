# Bottle Cap Mosaic Maker
Automatically create custom mosaics from your collection of bottle caps<br>
Just provide some framing details and an image<br>
Works for tabletop designs or framed wall pieces!<br>

![Conversion Progression Example](/shields.png)

Generated output info:

Alignment: staggered<br>
Length: 48 inches<br>
Width: 45 inches<br>
Columns: 31<br>
Rows: 42<br>
X_padding: 60.0 mm<br>
Y_padding: 7.23 mm<br>
Total Caps: 1281<br>
Weight: 6.21 lbs (2.82 kg)<br>
Actual Height: 47.4 inches<br>
Actual Width: 40.3 inches<br>


# Usage
Currently based on Python 3.10

Install dependencies
```
pip install -r requirements.txt
```
Customize config.ini<br>
```
IMAGE_NAME = shield.png

# Image will fit to minimum axis and pad appropriately, preserving scale
FRAME_WIDTH_INCHES = 45
FRAME_HEIGHT_INCHES = 48

# Add some distance from the edge of the frame
FRAME_MARGIN_INCHES = 0

# Integer space between each cap
CAP_BUFFER_MM = 3

# staggered or grid
ALIGNMENT = staggered
```
Run
```
python -m main
```
