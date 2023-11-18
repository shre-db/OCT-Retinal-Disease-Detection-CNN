import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import sys


class CustomDataset(Dataset):
    def __init__(self, class_name, root_dir, split='train', transform=None, test_size=0.05, val_size=0.05, random_seed=42):
        self.class_name = class_name
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of all image files in the folder
        self.image_files = [f for f in os.listdir(os.path.join(self.root_dir, self.class_name)) if f.endswith('.jpeg')]

        # Split the dataset into train, validation, and test sets
        self.train_images, self.test_images = train_test_split(self.image_files, test_size=test_size, random_state=random_seed)
        self.train_images, self.val_images = train_test_split(self.train_images, test_size=val_size, random_state=random_seed)

        if split == 'train':
            self.image_files = self.train_images
        elif split == 'val':
            self.image_files = self.val_images
        elif split == 'test':
            self.image_files = self.test_images
        else:
            raise ValueError("Invalid split. Use 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.class_name, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


root_dir = "D:/projects/healthcare/rscbjbr9sj-3/CellData/OCT/train_test_merged/categories"

train_cnv = CustomDataset(root_dir=root_dir, class_name='CNV', split='train', transform=None)
train_dme = CustomDataset(root_dir=root_dir, class_name='DME', split='train', transform=None)
train_drusen = CustomDataset(root_dir=root_dir, class_name='DRUSEN', split='train', transform=None)
train_normal = CustomDataset(root_dir=root_dir, class_name='NORMAL', split='train', transform=None)

val_cnv = CustomDataset(root_dir=root_dir, class_name='CNV', split='val', transform=None)
val_dme = CustomDataset(root_dir=root_dir, class_name='DME', split='val', transform=None)
val_drusen = CustomDataset(root_dir=root_dir, class_name='DRUSEN', split='val', transform=None)
val_normal = CustomDataset(root_dir=root_dir, class_name='NORMAL', split='val', transform=None)

test_cnv = CustomDataset(root_dir=root_dir, class_name='CNV', split='test', transform=None)
test_dme = CustomDataset(root_dir=root_dir, class_name='DME', split='test', transform=None)
test_drusen = CustomDataset(root_dir=root_dir, class_name='DRUSEN', split='test', transform=None)
test_normal = CustomDataset(root_dir=root_dir, class_name='NORMAL', split='test', transform=None)


def save_dataset_images(dataset, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for idx in range(len(dataset)):
        image = dataset[idx]
        class_name = dataset.class_name
        split = 'train' if dataset.image_files[idx] in dataset.train_images else 'val' if dataset.image_files[idx] in dataset.val_images else 'test'
        if class_name in ['CNV', 'DME', 'DRUSEN']:
            save_path = os.path.join(target_dir, split, 'ABNORMAL')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image.save(os.path.join(save_path, f"{class_name}_{split}_{idx}.jpg"))
        elif class_name == 'NORMAL':
            save_path = os.path.join(target_dir, split, 'NORMAL')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image.save(os.path.join(save_path, f"{class_name}_{split}_{idx}.jpg"))

        msg = 'Completed {}/{}'.format(idx+1, len(dataset))
        sys.stdout.write('\r' + msg)


        
target_dir = 'data/'

save_dataset_images(train_cnv, target_dir)
save_dataset_images(train_dme, target_dir)
save_dataset_images(train_drusen, target_dir)
save_dataset_images(train_normal, target_dir)

save_dataset_images(val_cnv, target_dir)
save_dataset_images(val_dme, target_dir)
save_dataset_images(val_drusen, target_dir)
save_dataset_images(val_normal, target_dir)

save_dataset_images(test_cnv, target_dir)
save_dataset_images(test_dme, target_dir)
save_dataset_images(test_drusen, target_dir)
save_dataset_images(test_normal, target_dir)
