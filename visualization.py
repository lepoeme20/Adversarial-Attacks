import os
import torch
import torchvision
import config
from utils import get_dataloader

def visualize():
    for idx, (images, _) in enumerate(dataloader):
        images = images.to(args.device)

        if args.vis_set_idx:
            if idx in args.vis_indices:
                imsave(images, idx)
        else:
            imsave(images, idx)

def imsave(images, idx):
    file_name = str(idx) + ".png"
    save_path = os.path.join(save_root, file_name)
    torchvision.utils.save_image(
        images, filename=save_path, nrow=args.vis_n_rows, normalize=True)

if __name__ == "__main__":
    args = config.get_config()

    if args.vis_normal:
        args.batch_size = args.vis_batch_size
        dataloader = get_dataloader(args)
        save_root = os.path.join("Adv_examples", args.dataset, "images/normal")
        os.makedirs(save_root, exist_ok=True)
    else:
        load_path = os.path.join("Adv_examples", args.dataset, args.attack_name + ".pt")
        adv_images, adv_labels = torch.load(load_path)

        adv_data = torch.utils.data.TensorDataset(adv_images, adv_labels)
        dataloader = torch.utils.data.DataLoader(
            adv_data, batch_size=args.vis_batch_size, shuffle=False)

        save_root = os.path.join("Adv_examples", args.dataset, "images", args.attack_name)
        os.makedirs(save_root, exist_ok=True)

    visualize()
