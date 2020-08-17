"""

Pre-train a network on Approximate Convex Decompisitions using a pairwise 
contrastive loss. 


Author: AruniRC
Date: Feb 2020
"""
import argparse
import os
import os.path as osp
from data_utils.ShapeNetDataLoader import PartNormalDataset, SelfSupPartNormalDataset, ACDSelfSupDataset
from tensorboard_logger import configure, log_value
import itertools
import torch
from torch import nn
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from test_acdfeat_modelnet import extract_feats, cross_val_svm
from sklearn.svm import LinearSVC
import time



DEBUG = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 
                'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 
                'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 
                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 
                'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 
                'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Pre-train on ACD Airplanes')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=100, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default=None, help='GPU to use [default: None]')
    parser.add_argument('--cudnn_off', action='store_true', default=False, help='disable CuDNN [default: False]')
    parser.add_argument('--seed', type=int, default=0, help='Seed for multiple runs [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    # parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg_ShapeNet', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='use normal information [default: False]')
    parser.add_argument('--l2_norm', action='store_true', default=False, help='unit-normalize features [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--rotation_z', action='store_true', default=False, help='use z-rotation jitter [default: False]')
    parser.add_argument('--rotation_z_45', action='store_true', default=False, help='use z-rotation 45 degrees [default: False]')
    parser.add_argument('--random_anisotropic_scale', action='store_true', default=False, help='anisotropic scaling jittering [default: False]')
    parser.add_argument('--modelnet_val', action='store_true', default=False, help='do validation on ModelNet40 [default: False]')
    parser.add_argument('--lr_clip', type=float,  default=1e-5, help='Minimum value of lr [default: 1e-5]')
    parser.add_argument('--dgcnn_k', type=int,  default=20, help='DGCNN k [default: 20]')
    # self-supervised loss setting
    parser.add_argument('--selfsup', action='store_true', default=True, help='use self-sup data [default: False]')
    parser.add_argument('--margin', type=float,  default=0.5, help='contrastive loss margin [default: 0.5]')
    parser.add_argument('--lmbda', type=float,  default=1.0, help='weight on self-sup loss [default: 1.0]')
    parser.add_argument('--n_cls_selfsup', type=int,  default=-1, help='self-sup samples per class [default: -1, all samples]')
    parser.add_argument('--ss_dataset', type=str, default='acd', help='self-sup dataset [default: dummy]')
    parser.add_argument('--ss_path', type=str, default='/srv/data2/mgadelha/ShapeNetACD/', help='self-sup dataset location [default: dummy]')
    parser.add_argument('--retain_overlaps', action='store_true', default=False, help='keep overlapping shapes with labeled data [default: False]')
    # few-shot setting
    parser.add_argument('--k_shot', type=int,  default=-1, help='few shot samples [default: -1, all samples]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CUDA ENV SETTINGS'''
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.cudnn_off:
        torch.backends.cudnn.enabled = False # needed on gypsum!


    # --------------------------------------------------------------------------
    '''CREATE DIR'''
    # --------------------------------------------------------------------------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pretrain_part_seg')
    experiment_dir.mkdir(exist_ok=True)
    dir_name = args.model + '_ShapeNet' + \
                '_k-%d_seed-%d_lr-%.6f_lr-step-%d_lr-decay-%.2f_wt-decay-%.6f_l2norm-%d' \
                % ( args.k_shot, args.seed, args.learning_rate, 
                    args.step_size, args.lr_decay, args.decay_rate, 
                    int(args.l2_norm) )
    if args.normal:
        dir_name = dir_name + '_normals'
    if args.selfsup:
        dir_name = dir_name + 'selfsup-%s_selfsup_margin-%.2f_lambda-%.2f' \
                    % (args.ss_dataset, args.margin, args.lmbda)
    if args.rotation_z:
        dir_name = dir_name + '_rotation-z'

    if args.rotation_z_45:
        dir_name = dir_name + '_rotation-z-45'

    if args.random_anisotropic_scale:
        dir_name = dir_name + '_aniso-scale'        

    experiment_dir = experiment_dir.joinpath(dir_name)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------------
    '''LOG'''
    # --------------------------------------------------------------------------
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    configure(log_dir) # tensorboard logdir
    log_string('OUTPUT DIR: %s' % experiment_dir)


    # --------------------------------------------------------------------------
    '''DATA LOADERS'''
    # --------------------------------------------------------------------------
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TRAIN_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='trainval', 
                                      normal_channel=args.normal, k_shot=args.k_shot)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, 
                                                  shuffle=True, num_workers=4)
    trainDataIterator = iter(trainDataLoader)

    TEST_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='test', 
                                     normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    if args.selfsup:
        log_string('Use self-supervision - alternate batches')
        if not args.retain_overlaps:
            log_string('\tRemove overlaps between labeled and self-sup datasets')
            labeled_fns = list(itertools.chain(*TEST_DATASET.meta.values())) \
                            + list(itertools.chain(*TRAIN_DATASET.meta.values()))
        else:
            log_string('\tUse all files in self-sup dataset')
            labeled_fns = []

        if args.ss_dataset == 'dummy':
            log_string('Using "dummy" self-supervision dataset (rest of labeled ShapeNetSeg)')
            SELFSUP_DATASET = SelfSupPartNormalDataset(root = root, npoints=args.npoint, 
                                        split='trainval', normal_channel=args.normal, 
                                        k_shot=args.n_cls_selfsup, labeled_fns=labeled_fns)
        elif args.ss_dataset == 'acd':
            log_string('Using "ACD" self-supervision dataset (ShapeNet Seg)')
            ACD_ROOT = args.ss_path
            SELFSUP_DATASET = ACDSelfSupDataset(root = ACD_ROOT, npoints=args.npoint, 
                                                normal_channel=args.normal, 
                                                k_shot=args.n_cls_selfsup, 
                                                exclude_fns=labeled_fns, 
                                                use_val = True)
            log_string('\t %d samples' % len(SELFSUP_DATASET))
            selfsup_train_fns = list(itertools.chain(*SELFSUP_DATASET.meta.values()))            
            log_string('Val dataset for self-sup')
            SELFSUP_VAL = ACDSelfSupDataset(root = ACD_ROOT, npoints=args.npoint, 
                                        normal_channel=args.normal, class_choice='Airplane', 
                                        k_shot=args.n_cls_selfsup, use_val=False,
                                        exclude_fns=selfsup_train_fns + labeled_fns)
            log_string('\t %d samples' % len(SELFSUP_VAL))

        selfsupDataLoader = torch.utils.data.DataLoader(SELFSUP_DATASET, batch_size=args.batch_size, 
                                                        shuffle=True, num_workers=4)        
        selfsupIterator = iter(selfsupDataLoader)
        selfsupValLoader = torch.utils.data.DataLoader(SELFSUP_VAL, batch_size=args.batch_size, 
                                                        shuffle=False, num_workers=4)

    log_string('Load ModelNet dataset for validation')
    DATA_PATH = 'data/modelnet40_normal_resampled/'
    MN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.npoint, 
                                    split='train', normal_channel=args.normal)
    modelnetLoader = torch.utils.data.DataLoader(MN_DATASET, batch_size=args.batch_size, 
                                                    shuffle=True, num_workers=4)


    # --------------------------------------------------------------------------
    '''MODEL LOADING'''
    # --------------------------------------------------------------------------
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    if args.model == 'dgcnn':
        classifier = MODEL.get_model(num_part, normal_channel=args.normal, k=args.dgcnn_k).cuda()
    else:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    

    criterion = MODEL.get_loss().cuda()

    if args.selfsup:
        selfsupCriterion = MODEL.get_selfsup_loss(margin=args.margin).cuda()
        log_string("The number of self-sup data is: %d" %  len(SELFSUP_DATASET))


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)


    # --------------------------------------------------------------------------
    '''OPTIMIZER SETTINGS'''
    # --------------------------------------------------------------------------
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    # LEARNING_RATE_CLIP = 1e-5
    LEARNING_RATE_CLIP = args.lr_clip
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier)


    # --------------------------------------------------------------------------
    '''TRAINING LOOP'''
    # --------------------------------------------------------------------------
    best_val_loss = 99999
    global_epoch = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_loss = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        num_iters = len(selfsupDataLoader) # calc an epoch based on self-sup dataset

        for i in tqdm(list(range(num_iters)), total=num_iters, smoothing=0.9):
            '''applying self-supervised constrastive (pairwise) loss'''
            try:
                data_ss = next(selfsupIterator)
            except StopIteration:
                # reached end of this dataloader
                selfsupIterator = iter(selfsupDataLoader)
                data_ss = next(selfsupIterator)

            # DEBUG
            if DEBUG and i > 10:
                break

            points, label, target = data_ss  # (points: bs x 3 x n_pts, label: bs x 1, target: bs x n_pts)
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])

            if args.random_anisotropic_scale:
                points[:,:, 0:3] = provider.random_anisotropic_scale_point_cloud(points[:,:, 0:3], 
                                    scale_low=0.8, scale_high=1.25)

            # pts = torch.Tensor(points)
            # pts = pts.transpose(2,1)
            # np.save(osp.join(experiment_dir, 'pts.npy'), pts.cpu().numpy())

            if args.rotation_z:
                points[:,:, 0:3] = provider.rotate_point_cloud_y(points[:,:, 0:3])
            
            if args.rotation_z_45:   
                points[:,:, 0:3] = provider.rotate_point_cloud_y_pi4(points[:,:, 0:3])

            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            # np.save(osp.join(experiment_dir, 'pts_z-rot.npy'), points.cpu().numpy())
            # np.save(osp.join(experiment_dir, 'target.npy'), target.cpu().numpy())

            # for self-sup category label is always unknown, so always zeros:
            category_label = torch.zeros([label.shape[0], 1, num_classes]).cuda()

            optimizer.zero_grad()
            classifier = classifier.train()

            _, _, feat = classifier(points, category_label) # feat: [bs x ndim x npts]

            ss_loss = selfsupCriterion(feat, target) * args.lmbda
            ss_loss.backward()
            optimizer.step()
            mean_loss.append(ss_loss.item())
            log_value('selfsup_loss_iter', ss_loss.data, epoch*num_iters + i + 1)


        train_loss_epoch = np.mean(mean_loss)
        log_string('Self-sup loss is: %.5f' % train_loss_epoch)
        log_value('selfsup_loss_epoch', train_loss_epoch, epoch)

        # # # DEBUG: 
        # with torch.no_grad():
        #     sa3_wt = classifier.sa3.mlp_convs[2].weight.mean()
        #     log_string('SA3 avg wt is: %.5f' % sa3_wt.item())
        #     log_value('sa3_conv2_wt', sa3_wt.item(), epoch)


        '''validation after one epoch'''
        log_string('Validation: ACD on ShapeNet')
        with torch.no_grad():
            total_val_loss = 0
            for batch_id, (points, label, target) in tqdm(enumerate(selfsupValLoader), 
                                                          total=len(selfsupValLoader), 
                                                          smoothing=0.9):
                if DEBUG and i > 10:
                    break
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                category_label = torch.zeros([label.shape[0], 1, num_classes]).cuda()
                classifier = classifier.eval()
                _, _, feat = classifier(points, category_label)
                val_loss = selfsupCriterion(feat, target)
                total_val_loss += val_loss.data.cpu().item()
            avg_val_loss = total_val_loss / len(selfsupValLoader)
        log_value('selfsup_loss_val', avg_val_loss, epoch)


        '''(optional) validation on ModelNet40'''
        if args.modelnet_val:
            log_string('Validation: SVM on ModelNet40')
            with torch.no_grad():
                log_string('Extract features on ModelNet40')
                if args.model == 'pointnet_part_seg':
                    feat_train, label_train = extract_feats_pointnet(
                                                classifier, modelnetLoader, subset=0.5) 
                elif args.model == 'pointnet2_part_seg_msg':
                    feat_train, label_train = extract_feats(
                                                classifier, modelnetLoader, subset=0.5) 
                else:
                    raise ValueError
                log_string('Training data: %d samples, %d features' % feat_train.shape) 
                start_time = time.time()
                log_string('Training SVM on ModelNet40')
                svm, best_C, best_score = cross_val_svm(feat_train, label_train, c_min=100, 
                                                        c_max=501, c_step=20, verbose=False)
                elapsed_time = time.time() - start_time
            log_string('ModelNet val Accuracy: %f (elapsed: %f seconds)' % (best_score, elapsed_time))
            log_value('modelnet_val', best_score, epoch)

        # save every epoch
        if epoch % 5 == 0:
            savepath = str(checkpoints_dir) + ('/model_%03d.pth' % epoch)
            log_string('Saving model at %s' % savepath)
            state = {
                'epoch': epoch,
                'selfsup_loss': ss_loss.data,
                'val_loss': avg_val_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saved model.')

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving best model at %s'% savepath)
            state = {
                'epoch': epoch,
                'selfsup_loss': ss_loss.data,
                'val_loss': avg_val_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saved model.')

        log_value('train_lr', lr, epoch)
        log_value('train_bn_momentum', momentum, epoch)

        log_string('Epoch %d Self-sup train loss: %f  Val loss: %f ' % (epoch+1, 
                                                                        train_loss_epoch,
                                                                        avg_val_loss))

        global_epoch+=1


if __name__ == '__main__':
    args = parse_args()
    main(args)

