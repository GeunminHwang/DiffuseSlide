from PIL import Image

# Load the .tif image
tif_image_path = '/hdd_1/ubin108/MetaFormer/assets/images/real/input/139.tif'
image = Image.open(tif_image_path)

# Save the image as .png
png_image_path = '/hdd_1/ubin108/MetaFormer/assets/images/real/input/139.png'
image.save(png_image_path, 'PNG')