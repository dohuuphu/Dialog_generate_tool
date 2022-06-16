import os
import glob
import numpy as np
import shutil
import faiss
import cv2 as cv

from os.path import join
from FaceRecognition.variables import *

class Database():
    def __init__(self):
        self.id_persons = DB_ROOT
        self.storage = faiss.IndexFlatIP(DIMENSION)
        self.map_storage = []
        self.num_speaker = 0
        # self.init_database()

    def init_database(self):
        list_personFolder = os.listdir(DB_ROOT)
        if list_personFolder is not None:
            self.num_speaker = len(list_personFolder)
            for personFolder in list_personFolder:
                path_personFolder = os.path.join(DB_ROOT, personFolder)
                if self.item_exist(path_personFolder, 'txt'):
                    for emb in self.read_spkEmb(path_personFolder):
                        self.add_newEmb2storage(emb, path_personFolder.split('/')[-1])  

    def clean_database(self):
        self.storage = None
        self.map_storage = []
        self.num_speaker = 0
        self.storage = faiss.IndexFlatL2(DIMENSION)

        [ shutil.rmtree(i) for i in glob.glob(f'{DB_ROOT}/*')]

    # read, convert and append emb to list_emb    
    def read_spkEmb(self, personal_folders):
        list_emb = []
        for emb_path in os.listdir(personal_folders):
            if '.txt' not in emb_path:
                continue
            try:    
                emb = np.genfromtxt(join(personal_folders, emb_path), dtype='f')
                # emb_raw = open(join(personal_folders, emb_path), 'r').read().replace('\n','')
                # emb = np.array([float(i) for i in emb_raw.split(',')], dtype='f')
                
                # list_emb.append(torch.from_numpy(emb))     
                list_emb.append(emb)  
            except OSError as e:
                print(f'Error of emb: {personal_folders} {emb_path}')
        return list_emb

    # create folder and append new speaker to storage     
    def save_spkEmb(self, root_folder, emb, name = None, prefix = ''):
        name = str(self.num_speaker + 1) if name is None else name
        name_ = name.strip().lower()
        personalFolder = os.path.join(root_folder, name_)

        if not os.path.exists(personalFolder) or not self.item_exist(personalFolder, 'txt'):
            try:
                shutil.rmtree(personalFolder)
            except OSError as e:
                print("Error: %s : %s" % (personalFolder, e.strerror))

            os.mkdir(personalFolder)
            self.num_speaker += 1 # update number speaker

        if self.add_newEmb2storage(emb, name):
            return self.write_emb(emb, personalFolder, prefix)

        return False, None


    def add_newEmb2storage(self, emb, name):
        try:
            emb_ = emb if len(emb.shape) ==2 else np.expand_dims(emb, axis=0)
            self.storage.add(emb_)  # add new emb to storage
            self.map_storage.append(name)   #add new spaker's name to mapping       
            return True
        except OSError as e:
            print(f'add_newEmb2storage func {e}')
            return False

    def write_emb(self, emb, personalFolder, prefix =''):
        try:
            num_item = len(glob.glob(f'{personalFolder}/{prefix}_*.txt'))
            emb_path = os.path.join(personalFolder,f'{prefix}_{str(num_item + 1)}.txt')
            np.savetxt(emb_path, emb)
            return True, emb_path
        except OSError as e:
            print(f'Write emb func {e}')
            return False, None
    
    def format_emb(self, emb):
        items = []
        for item in str(emb).split(' '):
            if item != '':
                items.append(item.strip().replace(']', '').replace('[', '') )
        return ','.join(items)

    # Check folder contain any item or item with format
    def item_exist(self, folder, format = None):
        if format is None:
            return glob.glob(f'{folder}/*')

        return glob.glob(f'{folder}/*.{format}')
    
    # def get_listFolder(self):
    #     return list(self.storage.keys())
    
    def get_personalFolder(self, personalFolder):
        list_folder = [item for item in list(self.storage.keys())]
        for folder in list_folder:
            if personalFolder == folder:
                return folder
        return None
    
    # def folder_exist(self, cur_folder):
    #     for folder in self.get_listFolder():
    #         if cur_folder == folder:
    #             return True
    #     return False
       
    # def save_spkEmb(self, name, emb):
    #     folder_path  = os.path.join(DB_ROOT, name.strip().lower())
    #     self.append_storage(folder_path, emb)            
            
            
        
        
    
    