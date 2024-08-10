"""
Contains functionality for creating PyTorch data loaders and for image classification data.
"""
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_directory: str,
                       test_directory: str,
                       train_transform: torchvision.transforms.Compose,
                       test_transform: torchvision.transforms.Compose,
                       batch_size: int,
                       num_workers: int = NUM_WORKERS):
  """ Creates training and testing dataloaders.
  Takes in training and testing directory paths, turns them into PyTorch datasets and then into dataloaders.

  Args:
    train_directory: path to training directory
    test_directory: path to testing directory
    train_transform: compose of  transformations for our training data
    test_transform: compose of  transformations for our testing data
    batch_size: how many data samples we want in one batch
    num_workers: do not change unless neccessary

  Returns:
    Train and test batched dataloaders and class_names.

  Examples:
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dirm
                                                                        test_dir,
                                                                        data_transforms,
                                                                        BATCH_SIZE)
  """
  train_data = datasets.ImageFolder(train_directory,
                                    transform=train_transform)
  test_data = datasets.ImageFolder(test_directory,
                                   transform=test_transform)
  train_dataloader = DataLoader(train_data,
                                batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
  test_dataloader = DataLoader(test_data,
                               batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               pin_memory=True)
  class_names = train_data.classes
  return train_dataloader, test_dataloader, class_names
