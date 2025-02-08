import torch
import torchvision.models as models

def save_imagenet_resnet18_checkpoint(out_path='resnet18_imagenet.pth'):
    """Download torchvision pretrained resnet18 and save as a local checkpoint."""
    model = models.resnet18(pretrained=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved pretrained ResNet-18 to {out_path}")

if __name__ == "__main__":
    save_imagenet_resnet18_checkpoint()