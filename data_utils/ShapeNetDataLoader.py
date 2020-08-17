# *_*coding:utf-8 *_*
import os
import os.path as osp
import json
import warnings
import numpy as np
import random
import math
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


DEBUG = True



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                 npoints=2500, split='train', class_choice=None, normal_channel=False, k_shot=-1):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.k_shot = k_shot


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:            
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)            

            if self.k_shot > 0 and len(fns) > self.k_shot:
                fns = random.sample(fns, self.k_shot) # random few-shot samples                
                pass

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            ppoint_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)








class SelfSupPartNormalDataset(Dataset):
    def __init__(self, root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                 npoints=2500, split='train', class_choice=None, normal_channel=False, 
                 k_shot=-1, labeled_fns=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.k_shot = k_shot
        self.labeled_files = set([osp.basename(x) for x in labeled_fns])
        # assert len(labeled_fns) == self.k_shot

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:            
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = os.listdir(dir_point)
            fns = sorted(list(set(fns) - set(self.labeled_files)))  # remove files used as labeled data
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            if self.k_shot > 0:
                print('Subsampling self-supervised dataset.')
                fns = random.sample(fns, self.k_shot)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            ppoint_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)









class ACDSelfSupDataset(Dataset):
    def __init__(self, root = '/srv/data2/mgadelha/ShapeNetACD/', 
                 npoints=2500, class_choice=None, normal_channel=False, 
                 k_shot=-1, exclude_fns=[], splits=None, use_val=False):
        '''
            Expected self-supervised dataset folder structure:

                ROOT
                  |--- <sub-folder-1>
                  |     | -- af55f398af2373aa18b14db3b83de9ff.npy
                  |     | -- ff77ea82fb4a5f92da9afa637af35064.npy
                  |    ...
                  |
                  |--- <sub-folder-2>
                 ...

            The "subfolders" loosely correspond to "object categories", but can 
            be arbitrary. The code works with a single subfolder. However, it 
            does not work if there are no subfolders at all under the ROOT path.

        '''
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.k_shot = k_shot
        self.meta = {}
        subfolders = os.listdir(root)
        self.classes_original = dict(zip(subfolders, range(len(subfolders))))
        self.cat = self.classes_original
        self.use_val = use_val
        if len(exclude_fns) > 0:
            self.exclude_fns = [osp.basename(x) for x in exclude_fns]
        else:
            self.exclude_fns = []
        # self.classes = self.classes_original

        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, item)
            fns = [f for f in os.listdir(dir_point) if f.endswith('.npy')]
            num_all_fns = len(fns)
            if len(self.exclude_fns) > 0:
                # print('Removing overlaps with excluded files list . . .')
                fns = sorted(list(set([osp.splitext(osp.basename(f))[0] for f in fns])
                                - set(osp.splitext(osp.basename(f))[0] for f in self.exclude_fns)))
                # print('Removed %d overlapping samples' % (num_all_fns - len(fns)))

            NUM_SAMPLES = len(fns)
            
            # support for specifying a random subset of the self-sup data
            if self.k_shot > 0:
                print('Subsampling self-supervised dataset (%d samples).' % args.k_shot)
                fns = random.sample(fns, self.k_shot)

            if self.use_val:
                # we fix 80/20 train/val splits per category
                fns = random.sample(fns, math.floor(NUM_SAMPLES * 0.8))
            
            for fn in fns:
                token = (osp.splitext(osp.basename(fn))[0])
                self.meta[item].append(osp.join(dir_point, token + '.npy')) # NOTE: .npy files

        self.datapath = []
        for item in self.cat.keys():
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            ppoint_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.load(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]  # TODO: make sure the extra cols with normals exist
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]  # resample
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)






class MultiACDSelfSupDataset(Dataset):
    def __init__(self, root = '/srv/data2/mgadelha/ShapeNetACD/', 
                 npoints=2500, class_choice=None, normal_channel=False, 
                 k_shot=-1, exclude_fns=[], splits=None, use_val=False):
        '''
            Expected self-supervised dataset folder structure:

                ROOT
                  |--- <sub-folder-1>
                  |     | -- af55f398af2373aa18b14db3b83de9ff.npy
                  |     | -- ff77ea82fb4a5f92da9afa637af35064.npy
                  |    ...
                  |
                  |--- <sub-folder-2>
                 ...

            The "subfolders" loosely correspond to "object categories", but can 
            be arbitrary. The code works with a single subfolder. However, it 
            does not work if there are no subfolders at all under the ROOT path.

        '''
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.k_shot = k_shot
        self.meta = {}
        subfolders = os.listdir(root)
        self.classes_original = dict(zip(subfolders, range(len(subfolders))))
        self.cat = self.classes_original
        self.use_val = use_val
        if len(exclude_fns) > 0:
            self.exclude_fns = [osp.basename(x) for x in exclude_fns]
        else:
            self.exclude_fns = []
        # self.classes = self.classes_original

        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, item)
            fns = [f for f in os.listdir(dir_point) if f.endswith('.npy')]
            num_all_fns = len(fns)
            if len(self.exclude_fns) > 0:
                # print('Removing overlaps with excluded files list . . .')
                fns = sorted(list(set([osp.splitext(osp.basename(f))[0] for f in fns])
                                - set(osp.splitext(osp.basename(f))[0] for f in self.exclude_fns)))
                # print('Removed %d overlapping samples' % (num_all_fns - len(fns)))

            NUM_SAMPLES = len(fns)
            
            # support for specifying a random subset of the self-sup data
            if self.k_shot > 0:
                print('Subsampling self-supervised dataset (%d samples).' % args.k_shot)
                fns = random.sample(fns, self.k_shot)

            if self.use_val:
                # we fix 80/20 train/val splits per category
                fns = random.sample(fns, math.floor(NUM_SAMPLES * 0.8))
            
            for fn in fns:
                token = (osp.splitext(osp.basename(fn))[0])
                self.meta[item].append(osp.join(dir_point, token + '.npy')) # NOTE: .npy files

        self.datapath = []
        for item in self.cat.keys():
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            ppoint_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.load(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]  # TODO: make sure the extra cols with normals exist
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]  # resample
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


