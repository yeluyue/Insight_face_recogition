#for iccv challenge 2019
# 1. train data loader from folder
# 2. test data from image list
# 3. tensor with Normalize and RGB image divide/255

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from .read_image_list_io import DatasetFromList, read_image_list_test


def get_test_dataset(img_root_path, image_list_path):
    train_transform = transforms.Compose([
        # trans.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = DatasetFromList(img_root_path, image_list_path, default_loader, train_transform, None, read_image_list_test)
    img_num = ds.__len__()
    return ds, img_num


def get_test_list_loader(img_root_path, image_list_path, batch_size, num_workers):
    celeb_ds, img_num = get_test_dataset(img_root_path, image_list_path)
    ds = celeb_ds
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True,
                        num_workers=num_workers)
    return loader, img_num

