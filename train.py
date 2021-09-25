import glob
from torchvision import transforms
from dataset import LPSet

train_dir = 'datasets/ISTD/train/train_A/*.png'
train_list = glob.glob(train_dir)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
train_set = LPSet(train_list, train_transforms)

