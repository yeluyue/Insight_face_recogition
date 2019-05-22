from torch.utils.data import Dataset
from PIL import Image
import cv2

def read_image_list(root_path, image_list_path):
    f = open(image_list_path, 'r')
    data = f.read().splitlines()
    f.close()

    samples = []
    for line in data:
        sample_path = '{}/{}'.format(root_path, line.split(' ')[0])
        class_index = int(line.split(' ')[1])
        samples.append((sample_path, class_index))
    return samples


def read_image_list_test(root_path, image_list_path):
    f = open(image_list_path, 'r')
    data = f.read().splitlines()
    f.close()

    samples = []
    for line in data:
        sample_path = '{}/{}'.format(root_path, line.split(' ')[0])
        image_prefix = line.split(' ')[0]
        samples.append((sample_path, image_prefix))
    return samples


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        #return img.convert('BGR')

def opencv_loader(path):
    img = cv2.imread(path)
    return img


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)

class DatasetFromList(Dataset):
    """A generic data loader where the image list arrange in this way: ::

        class_x/xxx.ext 0
        class_x/xxy.ext 0
        class_x/xxz.ext 0

        class_y/123.ext 1
        class_y/nsdf3.ext 1
        class_y/asd932_.ext 1

    Args:
        root (string): Root directory path.
        image_list_path (string) : where to load image list
        loader (callable): A function to load a sample given its path.
        image_list_loader (callable) : A function to read image-label pair or image-image-prefix pair
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.<aimed to image>
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it. <aimed to label>

     Attributes:
        samples (list): List of (sample path, image_index) tuples
    """

    def __init__(self, root, image_list_path, loader ,
                 transform=None, target_transform=None, image_list_loader = read_image_list):
        samples = image_list_loader(root, image_list_path)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in image_list: " + image_list_path + "\n"))

        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
