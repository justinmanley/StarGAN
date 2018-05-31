from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
import csv

def get_label_names(filename):
	with open(filename, 'r') as celeba_csv:
		celeba_reader = csv.reader(celeba_csv, delimiter=' ')
		for i, row in enumerate(celeba_reader):
			if i == 1:
				return row
	raise ValueError("Expected %s to have a header row" % filename)

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        celeba_metadata = np.loadtxt(self.attr_path, skiprows = 2, dtype = str)
        _, D = celeba_metadata.shape

        celeba_image_filenames = celeba_metadata[:, 0]

        # Clipping turns -1 into 0.
        celeba_labels = np.clip(celeba_metadata[:, 1:D].astype(np.float32), 0, 1)

        celeba_label_names = np.array([name for name in get_label_names(self.attr_path) if name])

        selected_labels = [i for i, name in enumerate(celeba_label_names) if name in self.selected_attrs]

        num_labels = celeba_labels.shape[1]
        inverse_celeba_labels = np.logical_not(celeba_labels)

        # Assume that there is only one selected_attr.
        selected_attr, selected_attr_idx = self.selected_attrs[0], selected_labels[0]

        matches_selected_attr = np.nonzero(celeba_labels[:,selected_attr_idx])[0]
        does_not_match_selected_attr_full = np.nonzero(inverse_celeba_labels[:,selected_attr_idx])[0]
        num_matches_selected_attr = matches_selected_attr.shape[0]

        print('matches_selected_attr.shape', matches_selected_attr.shape)
        print('does_not_match_selected_attr_full', does_not_match_selected_attr_full.shape)
        
        does_not_match_selected_attr = np.random.choice(
            does_not_match_selected_attr_full, num_matches_selected_attr)
        
        selected_test, selected_train = matches_selected_attr[0:1000], matches_selected_attr[1000:num_matches_selected_attr]
        non_selected_test, non_selected_train = does_not_match_selected_attr[0:1000], does_not_match_selected_attr[1000:num_matches_selected_attr]

        test = np.hstack((selected_test, non_selected_test))
        train = np.hstack((selected_train, non_selected_train))

        print('test.shape', test.shape)
        print('train.shape', train.shape)
        
        np.random.shuffle(test)
        np.random.shuffle(train)

        for i in range(test.shape[0]):
            idx = test[i]
            self.test_dataset.append([celeba_image_filenames[idx], [celeba_labels[idx, selected_attr_idx]]])

        for i in range(train.shape[0]):
            idx = train[i]
            self.train_dataset.append([celeba_image_filenames[idx], [celeba_labels[idx, selected_attr_idx]]])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
