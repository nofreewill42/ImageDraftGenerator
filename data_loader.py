
import torch
import pathlib
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T


def create_dummy_palette_image(palette):
    """
    Creates a dummy palette image with the given palette colors.
    """
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette([val for sublist in palette for val in sublist] + [0, 0, 0]*(256-len(palette)))
    return palette_img


def ensure_rgb_or_l_mode(img):
    """
    Ensures that the given image is in RGB or L mode.
    """
    if img.mode not in ["RGB", "L"]:
        return img.convert("RGB")
    return img


class PaletteConverter:
    def __init__(self):
        divisions = [0, 51, 102, 153, 204, 255]
        self.palette = [(r,g,b) for r in divisions for g in divisions for b in divisions]
        self.dummy_palette_image = create_dummy_palette_image(self.palette)
    
    def rgb2palette(self, img: PIL.Image) -> torch.tensor:
        """
        Converts the given image to a specified palette.

        Args:
            img (PIL.Image): Image to convert.

        Returns:
            torch.tensor: Converted image.
        """
        img = ensure_rgb_or_l_mode(img)
        img = img.quantize(palette=self.dummy_palette_image)
        # Convert to tensor
        img = np.array(img, dtype=np.uint8)  # uint8 because 6*6*6 = 216 < 256  ; (web-safe palette has 216 colors)
        img = torch.tensor(img, dtype=torch.long)  # long because cross-entropy loss expects long tensors
        return img
    
    def palette2rgb(self, img: torch.tensor) -> PIL.Image:
        """
        Converts the given palette image to a RGB image.

        Args:
            img (torch.tensor): palette image to convert.

        Returns:
            PIL.Image: RGB image.
        """
        # tensor to numpy
        img = img.cpu().numpy()
        img = np.array(img, dtype=np.uint8)
        # palette to rgb
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rgb[i,j] = self.palette[img[i,j]]
        # numpy to pil
        rgb = Image.fromarray(rgb)
        return rgb



class ImagePaletteDataset(torch.utils.data.Dataset):
    """
    Dataset that returns the palette image for the given image.
    In [1x1,2x2,4x4,8x8,16x16,32x32] sizes.
    """
    def __init__(self, img_paths, sizes=[1,2,4,8,16,32]):  # TODO: Add 64,128,256 sizes later
        self.img_paths = img_paths
        self.sizes = sorted(sizes, reverse=True)  # [32,16,8,4,2,1]
        self.palette_converter = PaletteConverter()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        raw_image = Image.open(img_path)
        images = []
        image = raw_image
        for size in self.sizes:
            # Resize and crop image
            image = T.Resize(size)(image)#raw_image)
            image = T.CenterCrop(size)(image)
            # Convert to palette
            image_tensor = self.palette_converter.rgb2palette(image)
            images.append(image_tensor)
        images = images[::-1]  # reverse the list because we want the smallest image first
        return images




def save_gif(image_tensors, save_name):
    """
    Creates a gif from the given image tensors.
    """
    images = []
    for image_tensor in image_tensors + image_tensors[1:-1][::-1]:
        image = dataset.palette_converter.palette2rgb(image_tensor)
        image = image.resize((256, 256), Image.NEAREST)
        images.append(image)
    images[0].save(f'images/{save_name}', save_all=True, append_images=images[1:], duration=150, loop=0)


if __name__ == '__main__':
    img_path = pathlib.Path('images/Dog.jpeg')
    img_paths = [img_path]
    dataset = ImagePaletteDataset(img_paths)
    images = dataset[0]

    save_gif(images, 'raw_image.gif')
