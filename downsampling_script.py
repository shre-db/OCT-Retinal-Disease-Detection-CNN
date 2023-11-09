import os
import random
# random.seed(0)
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageFilter
import sys

# Histogram Equalization
def apply_hist_eqlz(image):
  return ImageOps.equalize(image)

# Min-Max Scaling
class Normalize01:
  def __call__(self, img):
    return img.float().div(255)

# Denosing
class MedianFilter(object):
  def __init__(self, size=3) -> None:
    self.size=size
  def __call__(self, img):
    return img.filter(ImageFilter.MedianFilter(size=self.size))


# Transformations
transformations = T.Compose([
    T.Grayscale(),
    T.Resize((224, 224)),
    T.CenterCrop(224),
    # apply_hist_eqlz,  # Histogram Equalization
    MedianFilter(size=5),  # Denoising
    # T.ToTensor(),
    # Normalize01(),
])


# Create a custom dataset
class CustomDataset(Dataset):
  def __init__(self, num_samples, class_name, dataset='train', transform=None):
    self.num_samples = num_samples
    self.class_name = class_name
    self.dataset = dataset
    self.transform = transform

    class_images = os.listdir(os.path.join(self.dataset, self.class_name))

    # Randomly sample num_sample images from pnuemonia_images
    self.class_samples = random.sample(class_images, num_samples)

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    img_name = os.path.join(self.dataset, self.class_name, self.class_samples[idx])
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    return image
  
train_num_samples = 2000  # (2000 CNV + 2000 DME + 2000 DRUSEN) + (2000 * 3 = 6000 NORMAL) = 12000 training images
test_num_samples = 83  # (83 CNV + 83 DME + 83 DRUSEN) + (83 * 3 = 249 NORMAL) = 498 test images

def save_downsampled(dataset_type: 'str'):
    class_id = 1
    for dataset, class_name in [(cnv_images, 'ABNORMAL'), (dme_images, 'ABNORMAL'), (drusen_images, 'ABNORMAL'), (normal_images, 'NORMAL')]:
        for i, image in enumerate(dataset, 1):
            class_dir = os.path.join(target_dir, dataset_type, class_name)
            os.makedirs(class_dir, exist_ok=True)
            image.save(os.path.join(class_dir, f'{class_id}_{class_name}_{i}.jpg'))
            msg = 'Completed {}/{} - {}/{}'.format(i, len(dataset), class_id, 4)
            sys.stdout.write('\r' + msg)
        class_id += 1
        

cnv_images = CustomDataset(train_num_samples, 'CNV', 'train', transform=transformations)
dme_images = CustomDataset(train_num_samples, 'DME', 'train', transform=transformations)
drusen_images = CustomDataset(train_num_samples, 'DRUSEN', 'train', transform=transformations)
normal_images = CustomDataset(train_num_samples * 3, 'NORMAL', 'train', transform=transformations)

target_dir = 'downsampled_datasets'
save_downsampled('train')

print(" Training set downsampled ✔")

cnv_images = CustomDataset(test_num_samples, 'CNV', 'test', transform=transformations)
dme_images = CustomDataset(test_num_samples, 'DME', 'test', transform=transformations)
drusen_images = CustomDataset(test_num_samples, 'DRUSEN', 'test', transform=transformations)
normal_images = CustomDataset(test_num_samples * 3, 'NORMAL', 'test', transform=transformations)

save_downsampled('test')

print(" Test set downsampled ✔")
print("Downsampling complete ✔")
