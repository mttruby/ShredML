import torch
from torch.utils.data import DataLoader
from ml.nodes.trick_data.trick_data import TrickDataset


# handle the issue with empty image dir (happens for Ollie83 and Ollie91 at least)
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.utils.data._utils.collate.default_collate(batch)

def get_data_loader(for_dir: str = "outputs/cropped_images", batch_size: int = 8, shuffle: bool = False):
    print(for_dir)
    import os
    print(os.listdir(for_dir))
    dataset = TrickDataset(
    for_dir,
    {"Ollie":0, "Kickflip":1}#, "Other":2}
    )

    return DataLoader(dataset, batch_size=8, shuffle=shuffle, collate_fn=collate_fn)



if __name__ == "__main__":
    pass
