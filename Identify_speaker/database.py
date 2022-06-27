import os
from os.path import join
import glob
from variables import *
import logging
import numpy as np
import torch
import shutil


class Database():
    def __init__(self):
        self.id_persons = DB_ROOT
        self.storage = self.parse_database() # use for verify speaker
    
    def parse_database(self):
        storage = {}
        list_personFolder = os.listdir(DB_ROOT)
        if list_personFolder is not None:
            for personFolder in list_personFolder:
                path_personFolder = os.path.join(DB_ROOT, personFolder)
                if self.item_exist(path_personFolder):
                    storage.update({path_personFolder : self.read_spkEmb(path_personFolder)})
        return storage      
                
    def read_spkEmb(self, personal_folders):
        list_emb = []
        for emb_path in os.listdir(personal_folders):
            try:
                emb_raw = open(join(personal_folders, emb_path), 'r').read().replace('\n','')
                emb = np.array([float(i) for i in emb_raw.split(',')])
                list_emb.append(torch.from_numpy(emb))            
            except:
                print(f'Error of emb: {emb_path}')
        return list_emb
                
    def save_spkEmb(self, name, emb):
        personalFolder  = os.path.join(DB_ROOT, name.strip().lower())
        emb_ = emb.cpu().numpy()
        if not self.folder_exist(personalFolder) or not self.item_exist(personalFolder):
            # creaete folder and append new spkeaker to storage
            try:
                shutil.rmtree(personalFolder)
            except OSError as e:
                print("Error: %s : %s" % (personalFolder, e.strerror))

            os.mkdir(personalFolder)
            self.storage.update({personalFolder : [torch.from_numpy(emb_)]})
        else:
            # append new emb to exist speaker 
            self.storage[personalFolder].append(torch.from_numpy(emb_))

        self.write_emb(personalFolder, emb_)
        # print(self.storage)
        
    def write_emb(self, personalFolder, emb):
        num_item = len(os.listdir(personalFolder))
        emb_path = os.path.join(personalFolder,str(num_item + 1) + '.txt')
        with open(emb_path, 'w') as write:
            write.writelines(self.format_emb(emb))
            logging.info(f"Write new speaker embedding at {emb_path}")
    
    def format_emb(self, emb):
        items = []
        for item in str(emb).split(' '):
            if item != '':
                items.append(item.strip().replace(']', '').replace('[', '') )
        return ','.join(items)

    def item_exist(self, folder):
        return glob.glob(f'{folder}/*.txt')
    
    def get_listFolder(self):
        return list(self.storage.keys())
    
    def get_personalFolder(self, personalFolder):
        list_folder = [item for item in list(self.storage.keys())]
        for folder in list_folder:
            if personalFolder == folder:
                return folder
        return None
    
    def folder_exist(self, cur_folder):
        for folder in self.get_listFolder():
            if cur_folder == folder:
                return True
        return False
       
    # def save_spkEmb(self, name, emb):
    #     folder_path  = os.path.join(DB_ROOT, name.strip().lower())
    #     self.append_storage(folder_path, emb)            
            
            
        
        
    
    