# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import logging
from datasets.sixclass import OCT
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob as glob

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import topk
from utils.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter

from utils.utils import save_model, write_log
from utils.focal_loss import FocalLoss
from utils.lr_scheduler import inv_lr_scheduler
from datasets import *
from models import *
from models.gradcamres import GradCamModel

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        logging.info(f'--> trainer: {self.__class__.__name__}')

        self.setup()
        self.build_datasets()
        self.build_models()
        self.resume_from_ckpt()
        
    def setup(self):
        self.start_ite = 0
        self.ite = 0
        self.best_acc = 0.
        self.tb_writer = SummaryWriter(self.cfg.TRAIN.OUTPUT_TB)

    def build_datasets(self):
        logging.info(f'--> building dataset from: {self.cfg.DATASET.NAME}')
        self.dataset_loaders = {}

        # dataset loaders
        if self.cfg.DATASET.NAME == '6class':
            dataset = OCT
        elif self.cfg.DATASET.NAME == 'binary':
            dataset = OCT
            print(f'Dataset {self.cfg.DATASET.NAME}  found, root: {self.cfg.DATASET.ROOT}, source : {self.cfg.DATASET.SOURCE}')
        else:
            raise ValueError(f'Dataset {self.cfg.DATASET.NAME} not found')

        self.dataset_loaders['source_train'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='train'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_SOURCE,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['source_test'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='val', trim=self.cfg.DATASET.TRIM),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.dataset_loaders['target_train'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='train'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TARGET,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['target_test'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='test'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.len_src = len(self.dataset_loaders['source_train'])
        self.len_tar = len(self.dataset_loaders['target_train'])
        logging.info(f'    source {self.cfg.DATASET.SOURCE}: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')
        logging.info(f'    target {self.cfg.DATASET.TARGET}: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def build_models(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        self.base_net = self.build_base_models()
        self.registed_models = {'base_net': self.base_net}
        parameter_list = self.base_net.get_parameters()
        self.model_parameters()
        self.build_optim(parameter_list)

    def build_base_models(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'pretrained': self.cfg.MODEL.PRETRAIN,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
        }
        print("name",basenet_name)
        print('---------------')
        #basenet = GradCamModel()
        basenet = eval(basenet_name)(**kwargs).cuda()
        #print(basenet)

        return basenet

    def model_parameters(self):
        for k, v in self.registed_models.items():
            logging.info(f'    {k} paras: '
                         f'{(sum(p.numel() for p in v.parameters()) / 1e6):.2f}M')

    def build_optim(self, parameter_list: list):
        self.optimizer = optim.SGD(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            momentum=self.cfg.OPTIM.MOMENTUM,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            nesterov=True
        )
        self.lr_scheduler = inv_lr_scheduler


    def resume_from_ckpt(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-last.pt')
        last_ckpt = 'exps/binBI_FO/ckpt/biop2Foveal_123/models-best.pt'
        #exps/binRB_BP/ckpt/revised_biobankjpg2Biop_123/models-best.pt'
        #'exps/binFO_BP/ckpt/Foveal2Biop_123/models-best.pt'
        #'exps/binBI_FO/ckpt/biop2Foveal_123/models-best.pt'
        #'exps/binRB_FO/ckpt/revised_biobankjpg2Foveal_123/models-best.pt'
        #'exps/binBI_FO/ckpt/biop2Foveal_123/models-best.pt'
        #'/home/n/nrb27/classification_project/UDA/exps/binBI_RB/ckpt/biop2revised_biobankjpg_123/models-best.pt'
        #'/home/n/nrb27/classification_project/UDA/exps/binRB_FO/ckpt/revised_biobankjpg2Foveal_123/models-best.pt'
        #'/home/n/nrb27/classification_project/UDA/exps/binFO_RB/ckpt/Foveal2revised_biobankjpg_123/models-best.pt'
        #last_ckpt = '/home/n/nrb27/classification_project/UDA/exps/bin/ckpt/Foveal2Biop_123/models-best.pt'
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt)
            for k, v in self.registed_models.items():
                v.load_state_dict(ckpt[k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_ite = ckpt['ite']
            self.best_acc = ckpt['best_acc']
            logging.info(f'> loading ckpt from {last_ckpt} | ite: {self.start_ite} | best_acc: {self.best_acc:.3f}')
        else:
            logging.info('--> training from scratch')
       
      

    def train(self):
        # start training
        for _, v in self.registed_models.items():
            v.train()
        # print(v)
        for self.ite in range(self.start_ite, self.cfg.TRAIN.TTL_ITE):
            # test
            if self.ite % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.ite != self.start_ite:
                self.base_net.eval()
                self.test()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                ite_rate=self.ite / self.cfg.TRAIN.TTL_ITE * self.cfg.METHOD.HDA.LR_MULT,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
            if self.ite % self.len_src == 0 or self.ite == self.start_ite:
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.ite % self.len_tar == 0 or self.ite == self.start_ite:
                iter_tar = iter(self.dataset_loaders['target_train'])

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
            self.one_step(data_src, data_tar)
            if self.ite % self.cfg.TRAIN.SAVE_FREQ == 0 and self.ite != 0:
                self.save_model(is_best=False, snap=True)


    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src['image'].cuda(), data_src['label'].cuda()
        
        outputs_all_src = self.base_net(inputs_src)  # [f, y]

        loss_cls_src = FocalLoss.forward(outputs_all_src[1], labels_src)#F.cross_entropy(outputs_all_src[1], labels_src)

        loss_ttl = loss_cls_src

        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
            # tensorboard
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_ttl': loss_ttl.item(),
            })

    def display(self, data: list):
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | lr: {self.current_lr:.5f} '
        # update
        for _str in data:
            log_str += '| {} '.format(_str)
        logging.info(log_str)

    def update_tb(self, data: dict):
        for k, v in data.items():
            self.tb_writer.add_scalar(k, v, self.ite)

    def step(self, loss_ttl):
        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()    



    def getcam(self):
        # input image
        #LABELS_file = 'imagenet-simple-labels.json'
        image_file = '/data/neuroretinal/UDA/binary/revised_biobankjpg/Normal/5492400_21017_0_0.jpg'

        # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
        net=self.base_net

        net.eval()

        # hook the feature extractor
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        net._modules.get('layer4').register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(net.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        def returnCAM(feature_conv, weight_softmax, class_idx):
            # generate the class activation maps upsample to 256x256
            size_upsample = (256, 256)
            bz, nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
            return output_cam


        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
            ])

        # load test image
        img_pil = Image.open(image_file)
        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0))
        net.eval()
        logit ,gr = net(img_variable)
        print(logit)
        # load the imagenet category list
        classes=['normal','Abnormal']
        


        h_x = F.sigmoid(logit).data.squeeze()
        print(f'h_x:{h_x}')
        #probs, idx = h_x.sort(0, True)
        #probs = probs.numpy()
        probs=h_x.numpy()
        #idx = idx.numpy()
        print(f"probs:{probs}")

        # output the prediction
        for i in range(0, 1):
            print('{:.3f} -> {}'.format(probs, classes[0]))

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, [0])

        # render the CAM and output
        print('output CAM.jpg for the top1 prediction: %s'%classes[0])
        img = cv2.imread(p)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('/home/n/nrb27/classification_project/folder/CM_no_UDA/CAM.jpg', result)
        return result



    def test(self):
        logging.info('--> testing on source_test')
        src_acc = self.test_func(self.dataset_loaders['source_test'], self.base_net,cm=False)
        logging.info('--> testing on target_test')
        tar_acc = self.test_func(self.dataset_loaders['target_test'], self.base_net,cm=True)
        is_best = False
        if tar_acc > self.best_acc:
            self.best_acc = tar_acc
            is_best = True

        # display
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | src_acc: {src_acc:.3f} | tar_acc: {tar_acc:.3f} | ' \
                  f'best_acc: {self.best_acc:.3f}'
        logging.info(log_str)

        # save results
        log_dict = {
            'I': self.ite,
            'src_acc': src_acc,
            'tar_acc': tar_acc,
            'best_acc': self.best_acc
        }
        write_log(self.cfg.TRAIN.OUTPUT_RESFILE, log_dict)

        # tensorboard
        self.tb_writer.add_scalar('tar_acc', tar_acc, self.ite)
        self.tb_writer.add_scalar('src_acc', src_acc, self.ite)

        self.save_model(is_best=is_best)


    
    def test_func(self, loader, model,cm):
        predlist=[] #torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=[] #torch.zeros(0,dtype=torch.long, device='cpu')
        wf = open('heidelberg_norm.txt',"a")
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 200, self.cfg.TRAIN.PRINT_FREQ)
            accs = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), accs.avg))
                data = iter_test.__next__()
                inputs, labels = data['image'].cuda(), data['label'].cuda()
                #print(f'labels: {labels}')
                outputs_all = model(inputs)  # [f, y, ...]
                
                outputs = outputs_all[1]
                #print(outputs.cpu().numpy(),labels.cpu().numpy())
                preds=[]
                for ele in outputs.cpu().numpy() :
                    preds.append(np.argmax(ele))
                
                if cm:
                    for ele in outputs.cpu().numpy() :
                        wf.write(str(list(ele))+','+str(np.argmax(ele)))
                        wf.write('\n')

                #print(f'preds:{preds}\n labels:{labels.cpu().numpy()}')
                #print(f'preds: {preds}')
                acc = accuracy(outputs, labels)[0]
                #print(f'acc:{acc}')
                accs.update(acc.item(), labels.size(0))
                predlist+= preds  #list(outputs.detach().cpu().numpy()) # torch.cat([predlist,outputs.view(-1).cpu()])
                lbllist+=  list(labels.cpu().numpy()) # torch.cat([lbllist,classes.view(-1).cpu()])
        print(f'accavg:{accs.avg}')
        print(f'accu:{accuracy_score(predlist,lbllist)}')
        print(predlist,lbllist)
        wf.write('END\n')
        wf.close()
        if cm:

            #print(f'cof_matrix: {confusion_matrix(predlist,lbllist)}')
            #conf=confusion_matrix(predlist,lbllist)
            cm=self.cm_analysis(lbllist, predlist,ymap=None, figsize=(10,10))
            #r,i = get_mislabels(lbllist,predlist)
            #print(i)

            '''cm.index.name = 'Actual'
            cm.columns.name = 'Predicted'
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(cm, annot=True,cmap='YlGnBu')
            plt.savefig('/home/n/nrb27/classification_project/folder/UDA_CM/dann/'+self.cfg.DATASET.NAME+"_CM.png")
        '''

        return accs.avg

    def cm_analysis(self, y_true, y_pred,  ymap=None, figsize=(10,10)):
        """
        Generate matrix plot of confusion matrix with pretty annotations.
        The plot image is saved to disk.
        args: 
        y_true:    true label of the data, with shape (nsamples,)
        y_pred:    prediction of the data, with shape (nsamples,)
        filename:  filename of figure file to save
        labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` dddd using scikit-learn models.
                 with shape (nclass,).
        ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
        figsize:   the size of the figure plotted.
        """
        #idxs=[]
        #imgname=[]
          #rid = [0]*len(list(y_pred))
        i=0
        im=0
        tstpth = os.path.join(f'dataset_map/binary', f'{self.cfg.DATASET.TARGET[0]}_valid_bin.txt')
        with open (tstpth, "r") as myfile:
            for line in myfile:
                  if y_true[i]!=y_pred[i]:
                    logging.info(f'{i : }{line[:-3]}')
                    im+=1
                    #print(f'{i : }{line[:-3]}')
                    #idxs.append(i)
                    #imgname.append(line[:-3])
                  i+=1
        print(f'{im}/{i} images are misclassified')

        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in labels]
        if self.cfg.DATASET.NAME == '6class':
            labels=['Atypical', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Normal']
        else:
            labels=['Abnormal','Normal']
        print(labels)
        #print(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
    
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=annot, fmt='', ax=ax,cmap='Blues')
        plt.save('Fullukbtest_mat.png')
        #plt.savefig('/home/n/nrb27/classification_project/folder/UDA_CM/dann/'+self.cfg.DATASET.NAME+'_'+self.cfg.DATASET.SOURCE[0]+'_' + self.cfg.DATASET.TARGET[0]+"_cmp_mat.png")
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm,annot=True,cmap='Blues')
        plt.save('Fullukbtest_hm.png')
        #plt.savefig('/home/n/nrb27/classification_project/folder/UDA_CM/dann/'+self.cfg.DATASET.NAME+'_'+self.cfg.DATASET.SOURCE[0]+'_' + self.cfg.DATASET.TARGET[0]+"_CM.png")
        
        #plt.show()
        return cm
    
    
    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'ite': self.ite,
            'best_acc': self.best_acc
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, ite=self.ite, is_best=is_best, snap=snap)