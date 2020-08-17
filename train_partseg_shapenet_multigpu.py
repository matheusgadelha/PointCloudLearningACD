"""
Author: AruniRC
Date: Feb 2019
"""
import argparse
import os
import os.path as osp
import data_utils
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


torch.backends.cudnn.enabled = False


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



def train_init_class(classifier, criterion, trainDataLoader, num_classes, num_part):
    """ Pre-train the classifier layer using logistic regression """
    optim = torch.optim.SGD(classifier.conv2.parameters(), lr=0.1, momentum=0.5)
    num_epoch = 500

    for epoch in range(num_epoch):
        print('Init Classifier: Epoch (%d/%d):' % (epoch + 1, num_epoch))
        mean_correct = []
        mean_loss = []
        for batch_id, (points, label, target) in enumerate(trainDataLoader):
            cur_batch_size, NUM_POINT, _ = points.size()
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            optim.zero_grad()

            classifier = classifier.eval()  # batch stats aren't updated

            '''applying supervised cross-entropy loss'''            
            seg_pred, trans_feat, feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            
            mean_correct.append(correct.item() / (cur_batch_size * NUM_POINT))

            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optim.step()
            mean_loss.append(loss.item())
            print('classifier: batch (%d/%s) Loss: %f Acc:%f' % (batch_id, 
                                                               len(trainDataLoader), 
                                                               loss.item(), 
                                                               mean_correct[-1]))
        
        log_value('init_cls_loss', np.mean(mean_loss), epoch)
        log_value('init_cls_acc', np.mean(mean_correct), epoch)

        # print('Epoch: %d Accuracy: %f' % (epoch+1, np.mean(mean_correct)))
    classifier = classifier.train()

    return classifier


def parse_args():
    parser = argparse.ArgumentParser('Train PointNet++ PartSeg Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default=None, help='GPU to use [default: None]')
    parser.add_argument('--cudnn_off', action='store_true', default=False, help='disable CuDNN [default: False]')
    parser.add_argument('--seed', type=int, default=0, help='Seed for multiple runs [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='use normal information [default: False]')
    parser.add_argument('--category', action='store_true', default=False, help='use category label information [default: False]')
    parser.add_argument('--l2_norm', action='store_true', default=False, help='unit-normalize features [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--dgcnn_k', type=int,  default=20, help='DGCNN k [default: 20]')
    parser.add_argument('--num_classes', type=int,  default=16, help='Number of shape classes [default: 16]')
    parser.add_argument('--num_parts', type=int,  default=50, help='Number of part classes [default: 50]')
    # self-supervised loss setting
    parser.add_argument('--selfsup', action='store_true', default=False, help='use self-sup data [default: False]')
    parser.add_argument('--margin', type=float,  default=0.5, help='contrastive loss margin [default: 0.5]')
    parser.add_argument('--lmbda', type=float,  default=10.0, help='weight on self-sup loss [default: 10]')
    parser.add_argument('--n_cls_selfsup', type=int,  default=-1, help='self-sup samples per class [default: -1, all samples]')
    parser.add_argument('--ss_dataset', type=str, default='acd', help='self-sup dataset [default: acd]')
    parser.add_argument('--ss_path', type=str, default='/srv/data2/mgadelha/ShapeNetACD/', help='self-sup dataset location [default: dummy]')
    parser.add_argument('--retain_overlaps', action='store_true', default=False, help='keep overlapping shapes with labeled data [default: False]')
    # annealing the weight of the self-sup loss (lambda)
    parser.add_argument('--anneal_lambda', action='store_true', default=False, help='anneal lambda value [default: False]')
    parser.add_argument('--anneal_step',type=int, default=5, help='anneal lambda epochs [default: 10]')
    parser.add_argument('--anneal_rate',type=float, default=0.5, help='anneal lambda epochs [default: 10]')
    # few-shot setting
    parser.add_argument('--k_shot', type=int,  default=-1, help='few shot samples [default: -1, all samples]')
    parser.add_argument('--pretrained', type=str, default=None, help='pre-trained model path [default: None]')
    parser.add_argument('--init_cls', action='store_true', default=False, help='pre-train classifier layers [default: False]')

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
    experiment_dir = experiment_dir.joinpath('part_seg_shapenet')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        # if args.k_shot > 0:
        dir_name = args.model + '_ShapeNet' + \
                    '_k-%d_seed-%d_lr-%.6f_lr-step-%d_lr-decay-%.2f_wt-decay-%.6f_l2norm-%d' \
                    % ( args.k_shot, args.seed, args.learning_rate, 
                        args.step_size, args.lr_decay, args.decay_rate, 
                        int(args.l2_norm) )
        if args.normal:
            dir_name = dir_name + '_normals'
        if args.category:
            dir_name = dir_name + '_category-label'
        if args.selfsup:
            dir_name = dir_name + '_selfsup-%s_margin-%.2f_lambda-%.2f' \
                        % (args.ss_dataset, args.margin, args.lmbda)
        if args.anneal_lambda:
            dir_name = dir_name + '_anneal-lambda_step-%d_rate-%.2f' \
                        % (args.anneal_step, args.anneal_rate)


        experiment_dir = experiment_dir.joinpath(dir_name)
        # else:
        #     experiment_dir = experiment_dir.joinpath(args.log_dir)

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
    log_string('PARAMETERS ...')
    log_string(args)
    configure(log_dir) # tensorboard logdir


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
    num_classes = args.num_classes
    num_part = args.num_parts

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
                                                exclude_fns=labeled_fns)

        selfsupDataLoader = torch.utils.data.DataLoader(SELFSUP_DATASET, 
                                                        batch_size=args.batch_size, 
                                                        shuffle=True, num_workers=4)        
        selfsupIterator = iter(selfsupDataLoader)
    
    


    # --------------------------------------------------------------------------
    '''MODEL LOADING'''
    # --------------------------------------------------------------------------
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    if 'dgcnn' in args.model:
        print('DGCNN params')
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

    if args.pretrained is None:
        # Default: load saved checkpoint from experiment_dir or start from scratch
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrained model from checkpoints')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier = classifier.apply(weights_init)
    else:
        # Path to a pre-trained model is provided (self-sup)
        log_string('Loading pretrained model from %s' % args.pretrained)
        start_epoch = 0
        ckpt = torch.load(args.pretrained)
        classifier.load_state_dict(ckpt['model_state_dict'])

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

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier)    


    # --------------------------------------------------------------------------
    ''' MODEL TRAINING '''
    # --------------------------------------------------------------------------
    best_acc = 0
    global_epoch = 0

    if args.pretrained is not None:
        if args.init_cls:
            # Initialize the last layer of loaded model using logistic regression
            classifier = train_init_class(classifier, criterion, trainDataLoader, num_classes, num_part)

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        ''' Adjust (anneal) self-sup lambda '''
        if args.anneal_lambda:
            lmbda = args.lmbda * (args.anneal_rate ** (epoch // args.anneal_step))
        else:
            lmbda = args.lmbda

        '''learning one epoch'''
        num_iters = len(trainDataLoader) # num iters in an epoch
        if args.selfsup:
            num_iters = len(selfsupDataLoader) # calc an epoch based on self-sup dataset

        for i in tqdm(list(range(num_iters)), total=num_iters, smoothing=0.9, desc='Training'):
            # ------------------------------------------------------------------
            #   SUPERVISED LOSS
            # ------------------------------------------------------------------
            try:
                data = next(trainDataIterator)
            except StopIteration:
                # reached end of this dataloader
                trainDataIterator = iter(trainDataLoader)
                data = next(trainDataIterator)

            points, label, target = data
            cur_batch_size, NUM_POINT, _ = points.size()
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            
            if args.category:
                category_label = to_categorical(label, num_classes).contiguous()
            else:
                category_label = torch.zeros([label.shape[0], 1, num_classes]).cuda()
            
            optimizer.zero_grad()
            classifier = classifier.train()            

            '''applying supervised cross-entropy loss'''
            seg_pred, trans_feat, feat = classifier(points.contiguous(), category_label)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            mean_correct.append(correct.item() / (cur_batch_size * NUM_POINT))
            loss = criterion(seg_pred, target, trans_feat)

            loss.backward()
            optimizer.step()

            # ------------------------------------------------------------------
            #   SELF-SUPERVISED LOSS
            # ------------------------------------------------------------------
            if args.selfsup:
                try:
                    data_ss = next(selfsupIterator)
                except StopIteration:
                    # reached end of this dataloader
                    selfsupIterator = iter(selfsupDataLoader)
                    data_ss = next(selfsupIterator)

                points, label, target = data_ss
                points = points.data.numpy()
                points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
                points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
                points = torch.Tensor(points)
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                # for self-sup category label is always unknown, so always zeros:
                category_label = torch.zeros([label.shape[0], 1, num_classes]).cuda()
                if args.normal:
                    # put dummy cols of zeros for normals in self-sup data
                    cur_batch_size, _, NUM_POINT = points.size()
                    points = points[:,0:3,:]
                    points = torch.cat([points, torch.zeros([cur_batch_size, 3, NUM_POINT]).cuda()], 1)
                
                optimizer.zero_grad()
                classifier = classifier.train()

                '''applying self-supervised constrastive (pairwise) loss'''
                _, _, feat = classifier(points, category_label)
                ss_loss = selfsupCriterion(feat, target) * lmbda
                ss_loss.backward()
                optimizer.step()

        # ----------------------------------------------------------------------
        #   Logging metrics after one epoch
        # ----------------------------------------------------------------------
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)       
        log_string('Supervised loss is: %.5f' % loss.data)
        log_value('train_loss', loss.data, epoch)

        if args.selfsup:
            log_string('Self-sup loss is: %.5f' % ss_loss.data)
            log_value('selfsup_loss', ss_loss.data, epoch)

        # save every epoch
        savepath = str(checkpoints_dir) + ('/model_%03d.pth' % epoch)
        log_string('Saving model at %s' % savepath)
        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        # log_string('Saved model.')
        log_value('train_acc', train_instance_acc, epoch)
        log_value('train_lr', lr, epoch)
        log_value('train_bn_momentum', momentum, epoch)
        log_value('selfsup_lambda', lmbda, epoch)

        global_epoch+=1



    # ----------------------------------------------------------------------
    #   Evaluation on test-set after completing training epochs
    # ----------------------------------------------------------------------
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), 
                                                      total=len(testDataLoader), 
                                                      smoothing=0.9, desc='Evaluation'):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            if args.category:
                category_label = to_categorical(label, num_classes).contiguous()
            else:
                category_label = torch.zeros([label.shape[0], 1, num_classes]).cuda()

            classifier = classifier.eval()
            seg_pred, _, _ = classifier(points, to_categorical(label, num_classes))
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)

            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
                # print('\nTest IOUS: %f' % np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f' % (
         epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['instance_avg_iou']))
        



if __name__ == '__main__':
    args = parse_args()
    main(args)

