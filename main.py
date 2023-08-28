
from PIL import Image
from pathlib import Path

from torchvision import transforms as T


import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from utils import generate_websafe_palette, convert_to_palette, ensure_rgb_or_l_mode


# Use this function to get the web-safe colors
web_safe_colors = generate_websafe_palette()
print(web_safe_colors)


image_path = Path('images/Dog.jpeg')
image = T.Resize(256)(Image.open(image_path))
image = T.CenterCrop(256)(image)

# Create images in different sizes
images = [T.Resize(size)(image) for size in [1,2,4,8,16,32,64,128,256]]

# Convert the image using the web-safe palette
converted_img = convert_to_palette(image, web_safe_colors)
converted_img.show()
