import os
import random
random.seed(0)
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageFilter
import sys


# Transformations
transformations = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop(224),
])


# Create a custom dataset
class CustomDataset(Dataset):
   
  data_path = "D:/projects/healthcare/rscbjbr9sj-3/CellData/OCT/data_binarized_class"

  def __init__(self, class_name, dataset='train', transform=None):
    self.num_samples = 0
    self.class_name = class_name
    self.dataset = dataset
    self.transform = transform    
    class_images = os.path.join(self.data_path, self.dataset, self.class_name)
    self.class_samples = os.listdir(class_images)

  def __len__(self):
    return len(self.class_samples)

  def __getitem__(self, idx):
    img_name = os.path.join(self.data_path, self.dataset, self.class_name, self.class_samples[idx])
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    return image


def save_sampled(dataset_type: 'str'):
    class_id = 1
    for dataset, class_name in [(normal_images, 'NORMAL'), (abnormal_images, 'ABNORMAL')]:
        for i, image in enumerate(dataset, 1):
            class_dir = os.path.join(target_dir, dataset_type, class_name)
            os.makedirs(class_dir, exist_ok=True)
            image.save(os.path.join(class_dir, f'{class_id}_{class_name}_{i}.jpg'))
            msg = 'Completed {}/{} - {}/{}'.format(i, len(dataset), class_id, 2)
            sys.stdout.write('\r' + msg)
        class_id += 1
        

normal_images = CustomDataset('NORMAL', 'train', transform=transformations)
abnormal_images = CustomDataset('ABNORMAL', 'train', transform=transformations)


target_dir = 'data'
save_sampled('train')

print(" Training set downsampled ✔")

normal_images = CustomDataset('NORMAL', 'test', transform=transformations)
abnormal_images = CustomDataset('ABNORMAL', 'test', transform=transformations)


save_sampled('test')

print(" Test set downsampled ✔")
print("Downsampling complete ✔")