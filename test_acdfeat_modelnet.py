"""

Extract features from ACD-self-supervised model and train a linear SVM for the 
ModelNet40 classification task


Usage
-----
python test_acdfeat_modelnet.py

Author: AruniRC
Date: Feb 2020
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset, ACDSelfSupDataset
import argparse
import numpy as np
import os
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
import sys
import importlib
from sklearn.svm import LinearSVC
import provider
import time



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# default location to load checkpoint (can be specified as args):
# LOG_DIR = 'log/pretrain_part_seg/pointnet2_part_seg_msg_ShapeNet_k--1_seed-2004_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00'

# LOG_DIR = 'log/pretrain_part_seg/2020-02-11_16-40'
# LOG_DIR = 'log/pretrain_part_seg/2020-02-11_16-42'
# LOG_DIR = 'log/pretrain_part_seg/2020-02-11_16-44'


# ACD v2
# LOG_DIR = 'log/part_seg_shapenet/pointnet2_part_seg_msg_ShapeNet_k--1_seed-2001_lr-0.001000_lr-step-20_lr-decay-0.50_wt-decay-0.000100_l2norm-0'

# ACD with random rotations ***(giving good results)***
# LOG_DIR = 'log/pretrain_part_seg/pointnet2_part_seg_msg_ShapeNet_k--1_seed-0_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00_rotation-z'


# ACD with 45 degree rotations
# LOG_DIR = 'log/pretrain_part_seg/pointnet2_part_seg_msg_ShapeNet_k--1_seed-1_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00'


# ACD on PointNet model
# LOG_DIR = 'log/pretrain_part_seg/pointnet_part_seg_ShapeNet_k--1_seed-0_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00_rotation-z'



# ACD LR-clip 1e-8 - select checkpoint
# LOG_DIR = 'log/pretrain_part_seg/pointnet2_part_seg_msg_ShapeNet_k--1_seed-6_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00_rotation-z'


# ACD LR-clip 1e-5 (default) - select checkpoint
# LOG_DIR = 'log/pretrain_part_seg/pointnet2_part_seg_msg_ShapeNet_k--1_seed-4_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00_rotation-z'


# ACD rot 45 (with SVM val)
# LOG_DIR = 'log/pretrain_part_seg/pointnet2_part_seg_msg_ShapeNet_k--1_seed-3_lr-0.001000_lr-step-1_lr-decay-0.50_wt-decay-0.000100_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00_rotation-z'


# DGCNN -- segm architecture, no wt decay
LOG_DIR = 'log/pretrain_part_seg/dgcnn_seg_ShapeNet_k--1_seed-9992_lr-0.001000_lr-step-20_lr-decay-0.50_wt-decay-0.000000_l2norm-0selfsup-acd_selfsup_margin-0.50_lambda-1.00_rotation-z'


DEBUG = False



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', 
                        help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='model checkpoint filename [default: None]')
    parser.add_argument('--batch_size', type=int, default=24, 
                        help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, 
                        help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, 
                        help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, 
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--sqrt', action='store_true', default=False, 
                        help='Whether to use sqrt normalization [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, 
                        help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--cross_val_svm', action='store_true', default=False, 
                        help='Whether to cross-validate SVM C [default: False]')
    parser.add_argument('--svm_c', type=float, default=220.0, 
                        help='Linear SVM `C` hyper-parameter [default: 220.0]')
    parser.add_argument('--val_svm', action='store_true', default=False, 
                        help='Whether to use test or val set for eval [default: False]')
    parser.add_argument('--svm_jitter', action='store_true', default=False, 
                        help='Whether to jitter data during SVM training [default: False]')
    parser.add_argument('--do_sa3', action='store_true', default=False, 
                        help='Use SA3 layer features of PointNet++ [default: False]')
    parser.add_argument('--random_feats', action='store_true', default=False, 
                        help='Use randomly initialized PointNet++ features [default: False]')
    return parser.parse_args()




def acd_feat_normalize(l1_points, l2_points, feat):
    l1_feat, l2_feat, fc_feat = l1_points.mean(2), l2_points.mean(2), feat.mean(2)
    # l2-normalize and sqrt-normalize the mean-pooled features:
    desc = torch.cat([F.normalize(l1_feat, dim=1, p=2),      # l1: bs x 128 x 512
                    F.normalize(l2_feat, dim=1, p=2),        # l2: bs x 256 x 128 
                    F.normalize(fc_feat, dim=1, p=2)], 1)    # feat: bs x 128 x npts
    return F.normalize(desc, dim=1, p=2).cpu().numpy()



def _signed_sqrt(x):
    return torch.mul(torch.sign(x),torch.sqrt(torch.abs(x)+1e-12))


def test(classifier, model, loader, num_class=40, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):

        points_orig, target = data
        target = target[:, 0]
        points_orig, target = points_orig.cuda(), target.cuda()
        
        model = model.eval()

        # vote_pool = torch.zeros(target.size()[0],num_class).cuda()
        vote_pool = []

        # Extract features and get predictions from SVM classifier
        points = points_orig.transpose(2, 1).cuda()
        category_label = torch.zeros([target.shape[0], 1, 16]).cuda() # for self-sup, category label is always zeros
        _, (l1_points, l2_points, l3_points), feat = model(points, category_label)
        desc = acd_feat_normalize(l1_points, l2_points, feat)
        pred = classifier.decision_function(desc)  # (n_samples x n_classes)
        vote_pool.append(torch.Tensor(pred).unsqueeze(-1).cuda())

        # If vote_num > 1, jitter original point cloud and aggregate predictions
        # for i in range(vote_num-1):
        for n in range(1, 8):
            points = points_orig.cpu().data.numpy()
            # Test-time jitter
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.rotate_point_cloud_z(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.rotate_point_cloud_y(points[:,:, 0:3])
            rotation_angle = n * np.pi / 4
            points[:,:, 0:3] = provider.rotate_point_cloud_y_by_angle(points[:,:, 0:3], rotation_angle)
            
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points = points.cuda()

            # extract features and predict
            _, (l1_points, l2_points, l3_points), feat = model(points, category_label)
            desc = acd_feat_normalize(l1_points, l2_points, feat)
            pred = classifier.decision_function(desc)  # (n_samples x n_classes)

            vote_pool.append(torch.Tensor(pred).unsqueeze(-1).cuda())

        # max-pool over orientations
        vote_pool = torch.cat(vote_pool, 2)
        pred_scores = torch.max(vote_pool, 2)[0]
        pred_choice = pred_scores.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc




def extract_feats(model, loader, do_sqrt=True, pool='mean', do_sa3=False, do_svm_jitter=False, subset=-1):
    """ Return labels and extracted features on ModelNet data using ACD-pre-trained model """
    with torch.no_grad():
        feat_set = []
        label_set = []
        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            target = target[:, 0]

            if do_svm_jitter:
                # rotate pt-cloud about "up" axis by n*pi/4, n = 1, 2, ... 7
                pt_list = [points]
                for n in range(1, 8):
                    rotation_angle = n * np.pi / 4
                    # Input: BxNx3 numpy array
                    pcd_temp = provider.rotate_point_cloud_y_by_angle(points.data.numpy(), rotation_angle)                    
                    pt_list.append(torch.Tensor(pcd_temp))
                points = torch.cat(pt_list, 0)
                target = target.repeat(8)

            points = points.transpose(2, 1)  # Bx3xN
            points, target = points.float().cuda(), target.long().cuda()
            category_label = torch.zeros([target.shape[0], 1, 16]).cuda() # for self-sup, category label is always zeros
            model = model.eval()

            # if batch_id % 50 == 0:
            #     np.save(osp.join(experiment_dir, 'modelnet_points_%d.npy' % batch_id), 
            #             points.cpu().numpy())
            #     target_rep = target.unsqueeze(1).repeat([1, args.num_point])
            #     np.save(osp.join(experiment_dir, 'modelnet_target_%d.npy' % batch_id), 
            #             target_rep.cpu().numpy())
            if subset > 0:
                if batch_id > len(loader) * subset:
                    break

            _, (l1_points, l2_points, l3_points), feat = model(points, category_label)            
            if pool == 'mean':
                l1_feat, l2_feat, fc_feat = l1_points.mean(2), l2_points.mean(2), feat.mean(2)
            elif pool == 'max':
                l1_feat, l2_feat, fc_feat = l1_points.max(2)[0], l2_points.max(2)[0], feat.max(2)[0]
                do_sqrt = False
            else:
                raise ValueError

            if do_sa3:
                l3_feat = l3_points.squeeze()

            if do_sqrt:                
                l1_feat, l2_feat, fc_feat = torch.sqrt(l1_feat), torch.sqrt(l2_feat), torch.sqrt(fc_feat)
                if do_sa3:
                    l3_feat = torch.sqrt(l3_feat)


            desc = torch.cat([F.normalize(l1_feat, dim=1, p=2),      # l1: bs x 128 x 512
                            F.normalize(l2_feat, dim=1, p=2),        # l2: bs x 256 x 128 
                            F.normalize(fc_feat, dim=1, p=2)], 1)    # feat: bs x 128 x npts

            if do_sa3:
                desc = torch.cat([desc, F.normalize(l3_feat, dim=1, p=2)], 1)
            
            feat_set.append(F.normalize(desc, dim=1, p=2).cpu())
            label_set.append(target.squeeze().cpu())

        feat_train = torch.cat(feat_set, 0).numpy()
        label_train = torch.cat(label_set, 0).numpy()

    return feat_train, label_train





def extract_feats_dgcnn(model, loader, do_sqrt=True, pool='mean'):
    """ Return labels and extracted features on ModelNet data using ACD-pre-trained DGCNN model """
    with torch.no_grad():
        feat_set = []
        label_set = []
        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            target = target[:, 0]
            points = points.transpose(2, 1)  # Bx3xN
            points, target = points.float().cuda(), target.long().cuda()
            category_label = torch.zeros([target.shape[0], 1, 16]).cuda() # NOTE: for self-sup, category label is always zeros
            model = model.eval()

            _, (x1, x2, x3), feat = model(points, category_label)
            
            if pool == 'mean':
                l1_feat, l2_feat, l3_feat, fc_feat = x1.mean(2), x2.mean(2), x3.mean(2), feat.mean(2)
            elif pool == 'max':
                l1_feat, l2_feat, l3_feat, fc_feat = x1.max(2)[0], x2.max(2)[0], x3.max(2)[0], feat.max(2)[0]
                do_sqrt = False
            else:
                raise ValueError

            if do_sqrt:                
                l1_feat, l2_feat, l3_feat, fc_feat = _signed_sqrt(l1_feat), _signed_sqrt(l2_feat), _signed_sqrt(l3_feat), _signed_sqrt(fc_feat)

            desc = torch.cat([F.normalize(l1_feat, dim=1, p=2),      # l1: bs x 64 x 512
                            F.normalize(l2_feat, dim=1, p=2),        # l2: bs x 64 x 128 
                            F.normalize(l3_feat, dim=1, p=2),        # l3: bs x 64 x 128 
                            F.normalize(fc_feat, dim=1, p=2)], 1)    # feat: bs x 128 x npts            

            feat_set.append(F.normalize(desc, dim=1, p=2).cpu())
            label_set.append(target.squeeze().cpu())

        feat_train = torch.cat(feat_set, 0).numpy()
        label_train = torch.cat(label_set, 0).numpy()

    return feat_train, label_train



def extract_feats_pointnet(model, loader, do_sqrt=True, pool='mean', do_global=False, subset=-1):
    """ Return labels and extracted features on ModelNet data using ACD-pre-trained PointNet """
    with torch.no_grad():
        feat_set = []
        label_set = []
        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            target = target[:, 0]
            points = points.transpose(2, 1)  # Bx3xN
            points, target = points.float().cuda(), target.long().cuda()
            category_label = torch.zeros([target.shape[0], 1, 16]).cuda() # for self-sup, category label is always zeros
            model = model.eval()

            # if batch_id % 50 == 0:
            #     np.save(osp.join(experiment_dir, 'modelnet_points_%d.npy' % batch_id), 
            #             points.cpu().numpy())
            #     target_rep = target.unsqueeze(1).repeat([1, args.num_point])
            #     np.save(osp.join(experiment_dir, 'modelnet_target_%d.npy' % batch_id), 
            #             target_rep.cpu().numpy())
            if subset > 0:
                if batch_id > len(loader) * subset:
                    break

            _, global_feat, feat = model.feat_extract(points, category_label)            
            
            if pool == 'mean':
                fc_feat = feat.mean(2)
            elif pool == 'max':
                fc_feat = feat.max(2)[0]
                do_sqrt = False
            else:
                raise ValueError

            if do_sqrt:
                fc_feat = torch.sqrt(fc_feat)

            if do_global:
                desc = torch.cat([F.normalize(global_feat, dim=1, p=2), 
                                F.normalize(fc_feat, dim=1, p=2)], 1)
            else:
                desc = fc_feat
            
            feat_set.append(F.normalize(desc, dim=1, p=2).cpu())
            label_set.append(target.squeeze().cpu())

        feat_train = torch.cat(feat_set, 0).numpy()
        label_train = torch.cat(label_set, 0).numpy()

    return feat_train, label_train



def get_svm_trainval_split(feat, label):
    """Create training and validation splits from full training data """
    n_samples = label.shape[0]
    sel = list(range(n_samples))
    np.random.shuffle(sel)  # Comment this out if assuming dataloader has shuffle=True
    n_train = int(np.ceil(0.8 * n_samples)) # create (random) val split
    feat_train = feat[sel[0:n_train], :]
    label_train = label[sel[0:n_train]]
    feat_val = feat[sel[n_train:], :]
    label_val = label[sel[n_train:]]
    assert len(set(sel[0:n_train]).intersection(set(sel[n_train:]))) == 0
    return feat_train, label_train, feat_val, label_val



def train_val_svm(feat, label, svm_c=100.0):
    """ Create validation set and train+eval using a given value of SVM C """
    feat_train, label_train, feat_val, label_val = get_svm_trainval_split(feat, label)
    classifier = LinearSVC(random_state=123, multi_class='ovr', C=svm_c)
    print(classifier)
    classifier.fit(feat_train, label_train)
    val_score = classifier.score(feat_val, label_val)
    print('\t Subset Train data: %d samples, %d features' % feat_train.shape)
    print('\t Subset Val data: %d samples, %d features' % feat_val.shape)
    return val_score, (feat_train, label_train), (feat_val, label_val)


def cross_val_svm(feat, label, c_min=100.0, c_max=551.0, c_step=20.0, verbose=True):
    """ Model selection on validation set """
    n_samples = label.shape[0]
    sel = list(range(n_samples))
    np.random.shuffle(sel)  # Comment this out if assuming dataloader has shuffle=True
    n_train = int(np.ceil(0.8 * n_samples)) # create (random) val split
    feat_train = feat[sel[0:n_train], :]
    label_train = label[sel[0:n_train]]
    feat_val = feat[sel[n_train:], :]
    label_val = label[sel[n_train:]]

    assert len(set(sel[0:n_train]).intersection(set(sel[n_train:]))) == 0

    best_score = 0
    best_model = None
    c = 1.0
    for c in np.arange(c_min, c_max, c_step):
        classifier = LinearSVC(random_state=123, multi_class='ovr', C=c, dual=False)
        classifier.fit(feat_train, label_train)
        val_score = classifier.score(feat_val, label_val)
        if verbose:
            print('C=%f, score: %f' % (c, val_score))
        if val_score > best_score:
            best_score = val_score
            best_model = classifier
            best_C = c
    if verbose:
        print('Validation Best C=%f, score: %f' % (best_C, best_score))

    # now training with the best C
    classifier = LinearSVC(random_state=123, multi_class='ovr', C=best_C, dual=False)
    classifier.fit(feat[sel,:], label[sel])

    return classifier, best_C, best_score


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = osp.join(args.log_dir, 'ModelNet40-eval')
    experiment_dir = experiment_dir + '_' + str(args.num_point)
    if args.sqrt:
        experiment_dir = experiment_dir + '_do-sqrt'
    if args.do_sa3:
        experiment_dir = experiment_dir + '_sa3-feats'
    if args.svm_jitter:
        experiment_dir = experiment_dir + '_svm-jitter'
        args.batch_size = (args.batch_size // 8) # 8x augmentation
    if args.random_feats:
        experiment_dir = experiment_dir + '_random-feats'
    if args.ckpt is not None:
        experiment_dir = experiment_dir + '_' + osp.splitext(args.ckpt)[0]

    os.makedirs(experiment_dir, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    log_string('Experiment dir: %s' % experiment_dir)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, 
                                       split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, 
                                      split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, 
                                                  shuffle=False, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=4)
    

    if DEBUG:
        # ShapeNet training data
        shapenet_root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
        SHAPENET_DATASET = PartNormalDataset(root = shapenet_root, npoints=args.num_point, 
                                             split='trainval', normal_channel=args.normal)
        shapenetDataLoader = torch.utils.data.DataLoader(SHAPENET_DATASET, 
                                                        batch_size=args.batch_size, 
                                                        shuffle=False, num_workers=4)
        ACD_ROOT = '/srv/data2/mgadelha/ShapeNetACD/'
        SELFSUP_DATASET = ACDSelfSupDataset(root = ACD_ROOT, npoints=args.num_point, 
                                            normal_channel=args.normal)
        selfsupDataLoader = torch.utils.data.DataLoader(SELFSUP_DATASET, 
                                                        batch_size=args.batch_size, 
                                                        shuffle=False, num_workers=4)  


    '''MODEL LOADING'''
    shapenet_num_class = 50  # 
    model_name = args.model
    MODEL = importlib.import_module(model_name)
    model = MODEL.get_model(shapenet_num_class, normal_channel=False).cuda()
    if not args.random_feats:
        log_string('Load ACD pre-trained model: %s' % args.log_dir)
        if args.ckpt is None:
            checkpoint = torch.load(str(args.log_dir) + '/checkpoints/best_model.pth')
        else:
            checkpoint = torch.load(str(args.log_dir) + '/checkpoints/'+args.ckpt)
        
        try:
            DATA_PARALLEL = False
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            DATA_PARALLEL = True
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.module
    else:
        log_string('Using randomly initialized %s as feature extractor' % model_name)


    # Extract features and save
    if not osp.exists(osp.join(experiment_dir, 'train-feats.npy')) or \
        not osp.exists(osp.join(experiment_dir, 'train-labels.txt')):

        log_string('Extract features ...')
        if args.model == 'pointnet_part_seg':
            feat_train, label_train = extract_feats_pointnet(model, trainDataLoader, 
                                                    do_sqrt=args.sqrt,
                                                    do_global=args.do_sa3)
            feat_test, label_test = extract_feats_pointnet(model, testDataLoader, 
                                                    do_sqrt=args.sqrt,
                                                    do_global=args.do_sa3)

        elif args.model == 'pointnet2_part_seg_msg':
            feat_train, label_train = extract_feats(model, trainDataLoader, 
                                                    do_sqrt=args.sqrt, 
                                                    do_sa3=args.do_sa3, 
                                                    do_svm_jitter=args.svm_jitter)
            feat_test, label_test = extract_feats(model, testDataLoader, 
                                                    do_sqrt=args.sqrt, 
                                                    do_sa3=args.do_sa3, 
                                                    do_svm_jitter=args.svm_jitter)

        elif args.model == 'dgcnn':
            pass
            # feat_train, label_train = extract_feats_dgcnn(model, trainDataLoader, 
            #                                         do_sqrt=args.sqrt)
            # feat_test, label_test = extract_feats_dgcnn(model, testDataLoader, 
            #                                         do_sqrt=args.sqrt)
        elif args.model == 'dgcnn_seg':
            feat_train, label_train = extract_feats_dgcnn(model, trainDataLoader, 
                                                    do_sqrt=args.sqrt)
            feat_test, label_test = extract_feats_dgcnn(model, testDataLoader, 
                                                    do_sqrt=args.sqrt)
        else:
            raise ValueError

        np.save(osp.join(experiment_dir, 'train-feats.npy'), feat_train)
        np.savetxt(osp.join(experiment_dir, 'train-labels.txt'), label_train)
        np.save(osp.join(experiment_dir, 'test-feats.npy'), feat_test)
        np.savetxt(osp.join(experiment_dir, 'test-labels.txt'), label_test)

    else:
        log_string('Loading pre-trained features')
        feat_train = np.load(osp.join(experiment_dir, 'train-feats.npy'))
        label_train = np.loadtxt(osp.join(experiment_dir, 'train-labels.txt'))
        feat_test = np.load(osp.join(experiment_dir, 'test-feats.npy'))
        label_test = np.loadtxt(osp.join(experiment_dir, 'test-labels.txt'))    


    # Train linear SVM (one-vs-rest) on features
    
    

    # Train+test SVM on validation *or* test set
    log_string('Training linear SVM ...')
    if args.val_svm:
        log_string('Total data: %d samples, %d features' % feat_train.shape)
        val_acc, _, _ = train_val_svm(feat_train, label_train, svm_c=args.svm_c)
        log_string('Validation Accuracy: %f' % val_acc)
    else:
        # SVM training on *all* training data
        log_string('Training data: %d samples, %d features' % feat_train.shape)
        t_0 = time.time()
        if args.cross_val_svm:
            classifier, best_C, best_score = cross_val_svm(feat_train, label_train)
        else:
            classifier = LinearSVC(random_state=123, multi_class='ovr', C=args.svm_c, dual=False)
            classifier.fit(feat_train, label_train)
        train_acc = classifier.score(feat_train, label_train)
        log_string('Train Accuracy: %f' % train_acc)
        t_1 = time.time()
        log_string('Time elapsed: %f' % (t_1 - t_0))
        # test performance
        test_acc = classifier.score(feat_test, label_test)
        log_string('Test Accuracy: %f' % test_acc)

    
    # with torch.no_grad():
    #     instance_acc, class_acc = test(classifier, model, testDataLoader, vote_num=args.num_votes)
    # log_string('Rotations 45 at test time: \nTest Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
