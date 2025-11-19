import os
from functools import partial
from collections import namedtuple
from glob import glob
import numpy as np
from PIL import Image
import logging
import torch
import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import DataParallel

### wget https://download.pytorch.org/models/vgg16-397923af.pth

Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])


class IPR():
    def __init__(self, batch_size=50, k=3, num_samples=10000, model=None):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        if model is None:
            cnn = models.vgg16(pretrained=True)

        # https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/5
        # extract 2nd FC ReLU

        # 아래 말고 다음과 같이 뽑아 낼 수 도 있음. content_targets = [A.detach() for A in vgg(content_image, content_layers)] 
        # 다음 URL 참조. https://github.com/leongatys/PytorchNeuralStyleTransfer
            cnn.classifier = nn.Sequential(*[cnn.classifier[i] for i in range(5)])
            cnn = DataParallel(cnn).cuda().eval()

            self.vgg16 = cnn
        else:
            self.vgg16 = model

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(subject)
        distance_real_fake = compute_pairwise_distances(self.manifold_ref.features, manifold_subject.features)

        precision = (distance_real_fake <
                    np.expand_dims(self.manifold_ref.radii, axis=1)
                    ).any(axis=0).mean()

        recall = (distance_real_fake <
                  np.expand_dims(manifold_subject.radii, axis=0)
                  ).any(axis=1).mean()

        density = (1. / float(self.k)) * (
                distance_real_fake <
                np.expand_dims(self.manifold_ref.radii, axis=1)
                ).sum(axis=0).mean()

        coverage = (distance_real_fake.min(axis=1) <
                self.manifold_ref.radii
                ).mean()

        # precision = compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision...')
        # recall = compute_metric(manifold_subject, self.manifold_ref.features, 'computing recall...')
        return precision, recall, density, coverage

    def compute_manifold_ref(self, path):
        self.manifold_ref = self.compute_manifold(path)

    def realism(self, image):
        '''
        args:
            image: torch.Tensor of 1 x C x H x W
        '''
        feat = self.extract_features(image)
        return realism(self.manifold_ref, feat)

    def compute_manifold(self, input):
        '''
        Compute manifold of given input
        args:
            input: path or images, same as above
        returns:
            Manifold(features, radii)
        '''
        # features
        if isinstance(input, str):
            if input.endswith('samples.npz'):
                feats = self.extract_features_from_npz(input)
            elif input.endswith('.npz'):  # input is precalculated file
                print(input)
                f = np.load(input)
                feats = f['feature']
                radii = f['radii']
                f.close()
                return Manifold(feats, radii)
            else:  # input is dir
                feats = self.extract_features_from_files(input)
        elif isinstance(input, partial):
            feats = self.extract_features_from_sample_function(input)
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extract_features(input)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)

                feats = self.extract_features(input)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self.extract_features(input)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self.extract_features_from_files(input)
            else:
                raise TypeError
        else:
            logging.info(type(input))
            raise TypeError

        # radii
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        return Manifold(feats, radii)
    
    def extract_features_from_sample_function(self, sample):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        desc = 'extracting features of %d images' % self.num_samples
        images, _ = sample()
        batch_size, _, height, width = images.shape
        num_batches = int(np.ceil(self.num_samples / batch_size))

        if height != 224 or width != 224:
            resize = partial(F.interpolate, size=(224, 224))
        else:
            def resize(x): return x

        features = []
        with torch.no_grad():
          for bi in range(num_batches):
              start = bi * self.batch_size
              end = start + self.batch_size
              batch , _ = sample()
              batch = resize(batch)
              feature = self.vgg16(batch.cuda())
              features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)
    
    def extract_features(self, images):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        desc = 'extracting features of %d images' % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            resize = partial(F.interpolate, size=(224, 224))
        else:
            def resize(x): return x

        features = []
        with torch.no_grad():
          for bi in range(num_batches):
              start = bi * self.batch_size
              end = start + self.batch_size
              batch = images[start:end]
              batch = resize(batch)
              feature = self.vgg16(batch.cuda())
              features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)
    
    def extract_features_from_npz(self, images):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        images = torch.Tensor(np.load(images)['imgs'])
        images = images[:10000]
        print(images.shape)
        desc = 'extracting features of %d images' % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            resize = partial(F.interpolate, size=(224, 224))
        else:
            def resize(x): return x

        features = []
        with torch.no_grad():
          for bi in range(num_batches):
              start = bi * self.batch_size
              end = start + self.batch_size
              batch = images[start:end]
              batch = resize(batch)
              feature = self.vgg16(batch.cuda())
              features.append(feature.cpu().data.numpy())
        return np.concatenate(features, axis=0)
    

    def extract_features_from_files(self, path_or_fnames):
        """
        Extract features of vgg16-fc2 for all images in path
        params:
            path_or_fnames: dir containing images or list of fnames(str)
        returns:
            A numpy array of dimension (num images, dims)
        """

        dataloader = get_custom_loader(path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples)
        num_found_images = len(dataloader.dataset)
        if num_found_images < self.num_samples:
            logging.info('WARNING: num_found_images(%d) < num_samples(%d)' % (num_found_images, self.num_samples))
        resize = partial(F.interpolate, size=(224, 224))
        features = []
        for batch in dataloader:
 
            print(batch.min(), batch.max())
            print(batch.mean())            

            print(batch.shape)
            batch = resize(batch)
            print(batch.min(), batch.max())
            print(batch.mean())            

            feature = self.vgg16(batch.cuda())
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def save_ref(self, fname):
        print('saving manifold to', fname, '...')
        np.savez_compressed(fname,
                            feature=self.manifold_ref.features,
                            radii=self.manifold_ref.radii)


def compute_pairwise_distances(X, Y=None):
    '''
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    '''
    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
    else:
        num_Y = Y.shape[0]
    X = X.astype(np.float64)  # to prevent underflow
    X_norm_square = np.sum(X**2, axis=1, keepdims=True)
    if Y is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
    X_square = np.repeat(X_norm_square, num_Y, axis=1)
    Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    diff_square = X_square - 2*XY + Y_square

    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        # logging.info('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
        #       min_diff_square)

    distances = np.sqrt(diff_square)
    return distances


def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k+1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


def compute_metric(manifold_ref, feats_subject, desc=''):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in range(num_subjects):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects


def is_in_ball(center, radius, subject):
    return distance(center, subject) < radius


def distance(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)


def realism(manifold_real, feat_subject):
    feats_real = manifold_real.features
    radii_real = manifold_real.radii
    diff = feats_real - feat_subject
    dists = np.linalg.norm(diff, axis=1)
    eps = 1e-6
    ratios = radii_real / (dists + eps)
    max_realism = float(ratios.max())
    return max_realism

  
class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.fnames = glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
            glob(os.path.join(root, '**', '*.png'), recursive=True)

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:   
            image = self.transform(image)
        print(f'ap {torch.mean(image):.2f}, {torch.min(image):.2f}, {torch.max(image):.2f}')
        return image

    def __len__(self):
        return len(self.fnames)


class FileNames(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


def get_custom_loader(image_dir_or_fnames, image_size=224, batch_size=64, num_workers=4, num_samples=-1):
    norm_mean = [0.5,0.5,0.5]
    norm_std = [0.5,0.5,0.5]
    transform = []
    transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    # transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                       std=[0.229, 0.224, 0.225]))
    # transform.append(transforms.Normalize(mean=norm_mean,
    #                                       std=norm_std))
    transform = transforms.Compose(transform)

    if isinstance(image_dir_or_fnames, list):
        dataset = FileNames(image_dir_or_fnames, transform)
    elif isinstance(image_dir_or_fnames, str):
        dataset = ImageFolder(image_dir_or_fnames, transform=transform)
    else:
        raise TypeError

    if num_samples > 0:
        dataset.fnames = dataset.fnames[:num_samples]
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return data_loader



# This produces a function which takes in an iterator which returns a set number of samples
# and iterates until it accumulates config['num_inception_images'] images.
# The iterator can return samples with a different batch size than used in
# training, using the setting confg['inception_batchsize']
def prepare_pr_metrics(config):
    dset = config['dataset'].split('_')[0]

    assert os.path.exists('samples/features/'+dset+'_vgg_features.npz'), (
                'Make sure you have launched calculate_vgg_features.py before ! '
                     )

    ipr = IPR(batch_size=64, k=5, num_samples=config['num_pr_images'], model=None)
    ipr.compute_manifold_ref('samples/features/'+dset+'_vgg_features.npz')

    def get_pr_metrics(sample, prints=False):
        if prints:
            logging.info('Gathering activations and computing pr')
        precision, recall, density,  coverage = ipr.precision_and_recall(sample)
    
        return precision, recall, density, coverage
    
    return get_pr_metrics


