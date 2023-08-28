
import torch
import pathlib
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T


def ensure_rgb_or_l_mode(img):
    if img.mode not in ["RGB", "L"]:
        return img.convert("RGB")
    return img


def convert_to_palette(img, palette_colors):
    """
    Converts the given image to a specified palette.
    """
    palette_img = Image.new("P", (1, 1))  # Dummy image
    palette_img.putpalette([val for sublist in palette_colors for val in sublist] + [0, 0, 0]*(256-len(palette_colors)))
    return img.quantize(palette=palette_img)


def generate_websafe_palette():
    """
    Generates the web-safe palette.
    """
    colors = []
    palette_divisions = [0, 51, 102, 153, 204, 255]
    for r in palette_divisions:
        for g in palette_divisions:
            for b in palette_divisions:
                colors.append((r, g, b))
    return colors






class ImagePaletteDataset(torch.utils.data.Dataset):
    """
    Dataset that returns the palette image for the given image.
    In [1x1,2x2,4x4,8x8,16x16,32x32] sizes.
    """
    def __init__(self, img_paths, sizes=[1,2,4,8,16,32,64,128,256]):  # TODO: Add 64,128,256 sizes later
        self.img_paths = img_paths
        self.sizes = sorted(sizes, reverse=True)  # [32,16,8,4,2,1]
        self.web_safe_colors = generate_websafe_palette()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        raw_image = Image.open(img_path)
        images = []
        image = raw_image
        for size in self.sizes:
            # TODO: compare iteratively resized images vs resized for each size from the original image - START; [] means it was a copilot suggestion
            # (1,2,4,8,16,32)
            #image = T.Resize(size)(image)  # Time [taken to load an image from the dataset: 0.00010013580322265625 seconds (averaged over 100 iterations)]
            image = T.Resize(size)(image)  # Time taken to load an image from the dataset: 0.012885210514068603 seconds (averaged over 100 iterations)
            #image = T.Resize(size)(raw_image)  # Time taken to load an image from the dataset: 0.013923454284667968 seconds (averaged over 100 iterations)
            # (1,2,4,8,16,32,64,128,256,512)
            # Time taken to load an image from the dataset: 0.024085109233856202 seconds
            # Time taken to load an image from the dataset: 0.05549515247344971 seconds
            # (1,2,4,8,16,32,64,128,256)
            # Time taken to load an image from the dataset: 0.01784039497375488 seconds
            # Time taken to load an image from the dataset: 0.04203600645065308 seconds
            # TODO: compare iteratively resized images vs resized for each size from the original image - END
            image = T.CenterCrop(size)(image)
            image = ensure_rgb_or_l_mode(image)
            image = convert_to_palette(image, self.web_safe_colors)
            #image = np.array(image)
            images.append(image)
        return 


if __name__ == '__main__':
    img_path = pathlib.Path('images/Dog.jpeg')
    img_paths = [img_path]
    dataset = ImagePaletteDataset(img_paths)
    print(len(dataset))
    # Measure how much time it takes to load an image from the dataset in average
    import time
    start = time.time()
    for i in tqdm(range(100)):
        dataset[0]
    end = time.time()
    print(f"Time taken to load an image from the dataset: {(end-start)/100} seconds")
    print(dataset[0])
    dataset[0].show()
    print(None)