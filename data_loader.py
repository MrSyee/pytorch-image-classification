import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import cv2

traindir = "./data/train"
batch_size = 8
# pretrain normalize: https://pytorch.org/vision/stable/models.html
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            # transforms.Lambda(shear),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]
    ),
)
print(type(dataset))
img = next(iter(dataset))[0].permute(1, 2, 0).numpy()
print(img.shape)
cv2.imwrite("img.png", img)


train_loader = data.DataLoader(
    datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                # transforms.Lambda(shear),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

for i, (images, target) in enumerate(train_loader):
    print(i)
    print(images.size())
    print(target.size())
    break
