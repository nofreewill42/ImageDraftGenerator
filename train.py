
from pathlib import Path
from tqdm import tqdm


import torch

from data_loader import ImagePaletteDataset
from model import Model


if __name__ == "__main__":

    # Device - START
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Device - END

    # Dataset - START
    img_path = Path('images/Dog.jpeg').absolute()
    img_paths = [img_path]
    dataset = ImagePaletteDataset(img_paths)
    # Dataset - END

    # Hyperparameters - START
    epochs_num = 1000
    batch_size = 1
    num_workers = 0
    shuffle = True
    lr = 1e-3
    total_steps = epochs_num * len(dataset) // batch_size
    # Hyperparameters - END

    # Dataloader - START
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # Dataloader - END

    # Model - START
    model = Model(216, 512, 8, 6, 2048, 0.1, "relu", True)
    model.to(device)
    # Model - END

    # Optimizer - START
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()
    # Optimizer - END

    # Training - START
    pbar = tqdm(range(epochs_num))
    for epoch in pbar:
        for i, batch in enumerate(dataloader):#tqdm(dataloader)):
            images = batch

            images = torch.cat([i.flatten() for i in images]).unsqueeze(0)
            images = images.to(device)

            input_palette = images[:, :-1]
            target_palette = images[:, 1:]

            # TODO: # Train 64 images up to 16x16 size at once - START
            # step one with scaler optimizer
            # and then up to 32x32 size but one by one in a for loop
            # with gradient accumulation
            # then step again
            # TODO: # - END
            with torch.cuda.amp.autocast():
                preds = model(input_palette)
                loss = torch.nn.functional.cross_entropy(preds.flatten(end_dim=1), target_palette.flatten())

            #print(loss.item())

            scaler.scale(loss).backward()
            #if i % 100 == 0:
            # clip gradients, update optimizer, zero gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            pbar.set_description(f'Epoch: {epoch+1}/{epochs_num} | Loss: {loss.item():.4f}')

    # Training - END

    # Save model - START
    torch.save(model.state_dict(), 'model.pth')
    # Save model - END
    print(None)

