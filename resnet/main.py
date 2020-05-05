"""Main module to training classifier (ResNet)
"""
from resnet import config, trainer

if __name__ == "__main__":
    args = config.get_config()
    print(args)

    trainer.train(args)
    print("Well Done.")
