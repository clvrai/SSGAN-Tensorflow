from __future__ import print_function
import os
import sys
import tarfile
import subprocess
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Download dataset for SSGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+', choices=['mnist', 'cifar10'])

def download_mnist(download_path):
    data_dir = os.path.join(download_path, 'mnist')
    if os.path.exists(data_dir):
        print('MNIST was downloaded.')
        # return
    else:
        os.mkdir(data_dir)
    data_url = 'http://yann.lecun.com/exdb/mnist/'
    keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    
    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    train_image = loaded[16:].reshape((60000,28,28)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((60000)).astype(np.float))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    test_image = loaded[16:].reshape((10000,28,28)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((10000)).astype(np.float))

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.uint8)

    print ('Preprocessing data...')

    import progressbar
    from time import sleep
    bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    data_id = open(os.path.join(data_dir,'id.txt'), 'w')
    for i in range(image.shape[0]):
        if i%(image.shape[0]/100)==0: 
            bar.update(i/(image.shape[0]/100)+1)
        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        grp['image'] = image[i]
        label_vec = np.zeros(10)
        label_vec[label[i]] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()

    for k in keys:
        cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
        subprocess.call(cmd)

def download_cifar10(download_path):
    data_dir = os.path.join(download_path, 'cifar10')
    if os.path.exists(data_dir):
        print('CIFAR10 was downloaded.')
        return
    else:
        os.mkdir(data_dir)
    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    k = 'cifar-10-binary.tar.gz'
    target_path = os.path.join(data_dir, k)
    print(target_path)
    cmd = ['curl', data_url, '-o', target_path]
    print('Downloading CIFAR10')
    subprocess.call(cmd)
    # tarfile.open(target_path, 'r:gz').extractall('.')

if __name__ == '__main__':
    args = parser.parse_args()
    path = './datasets'
    if not os.path.exists(path): os.mkdir(path)

    if 'mnist' in args.datasets:
        download_mnist('./datasets')
    if 'cifar10' in args.datasets:
        download_cifar10('./datasets')
