# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os

from datasets.common_dataset import CommonDataset
from datasets.reader import read_images_labels


class OCT(CommonDataset):
    """
    -data_root:
     |
     |-art
     |-clipart
     |-product
     |-real_world
       |-Alarm_Clock
         |-0001.jpg
    """
    
    def __init__(self, data_root, domains: list, status: str = 'train', trim: int = 0):
        super().__init__(is_train=(status == 'train'))
 
        self._domains = ['bbankjpg','Biop','biop','Foveal','revised_biobankjpg','biobankLEjpg']

        if domains[0] not in self._domains:
            raise ValueError(f'Expected \'domain\' in {self._domains}, but got {domains[0]}')
        _status = ['train', 'val', 'test']
        if status not in _status:
            raise ValueError(f'Expected \'status\' in {_status}, but got {status}')

        self.image_root = data_root

        # read txt files
        
        if status == 'train':
            '''if domains[0]=='biobank':
                pth = os.path.join(f'dataset_map/binary_biobank_train_bin.txt')
            else:
                pth = os.path.join(f'dataset_map/binary', f'{domains[0]}_train_bin.txt')'''
            
            if domains[0]  == 'biobankLEjpg':
                data = read_images_labels(
                #os.path.join(f'dataset_map/6class', f'{domains[0]}_train.txt'), #6class
                os.path.join(f'dataset_map/binary', f'revised_biobankjpg_train_bin.txt'), #binary f'dataset_map/binary', f'{domains[0]}_valid_bin.txt
                shuffle=False,
                trim=0)
            else:
                data = read_images_labels(
                #os.path.join(f'dataset_map/6class', f'{domains[0]}_train.txt'), #6class
                os.path.join(f'dataset_map/binary', f'{domains[0]}_train_bin.txt'), #binary
                shuffle=True,
                trim=0)
        else:
            if domains[0]  == 'biobankLEjpg':
                data = read_images_labels(
                #os.path.join(f'dataset_map/6class', f'{domains[0]}_train.txt'), #6class
                os.path.join(f'dataset_map/binary', f'biobankLEjpg.txt'), #binary f'dataset_map/binary', f'{domains[0]}_valid_bin.txt
                shuffle=False,
                trim=0)
            else : 
                data = read_images_labels(
                #os.path.join(f'dataset_map/6class', f'{domains[0]}_train.txt'), #6class
                os.path.join(f'dataset_map/binary', f'{domains[0]}_valid_bin.txt'), #binary f'dataset_map/binary', f'{domains[0]}_valid_bin.txt
                shuffle=False,
                trim=0
        )
        '''
        data = read_images_labels(
            #os.path.join(f'dataset_map/6class', f'{domains[0]}_train.txt'), #6class
            os.path.join(f'dataset_map/binary', f'{domains[0]}_train_bin.txt'),
            shuffle=(status == 'train'),
            trim=0
        )'''

        self.data = data
        self.domain_id = [0] * len(self.data)

    def __len__(self):
        return len(self.data)