
from tqdm import tqdm

import torch

from model import Model


if __name__ == '__main__':

    # Device - START
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Device - END

    # Load model state dict - START
    model = Model(216, 512, 8, 6, 2048, 0.1, "relu", True)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    model.to(device)
    # Load model state dict - END

    # Dummy input - START
    palettes = [86]
    input_palette = torch.tensor(palettes).to(device).unsqueeze(0)
    # Dummy input - END

    # Predict - START
    for i in tqdm(range(1**2+2**2+4**2+8**2+16**2+32**2)):
        with torch.no_grad():
            output_palette = model(input_palette)
            pred_palette = output_palette[0,-1].argmax().item()
            palettes.append(pred_palette)
            input_palette = torch.tensor(palettes).to(device).unsqueeze(0)
    # Predict - END
    
    print(None)