"""Main module to training classifier (ResNet)
"""
import os
import config
from utils import network_initialization, get_dataloader
from methods import pgd, deepfool, fgsm, cw

def attack():
    dataloader = get_dataloader(args)
    target_cls = network_initialization(args)
    attack_module = globals()[args.attack_name.lower()]
    attack_func = getattr(attack_module, args.attack_name)
    attacker = attack_func(target_cls, args)
    save_path = os.path.join("Adv_examples", args.dataset.lower())
    attacker.inference(
        data_loader=dataloader, save_path=save_path, file_name=args.attack_name+".pt"
        )

if __name__ == "__main__":
    args = config.get_config()
    print(args)

    attack()
    print("The process is done.")
