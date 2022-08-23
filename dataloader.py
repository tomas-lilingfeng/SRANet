import pandas as pd 
import numpy as np
import SimpleITK as sitk
import torch.utils.data as data_utils
import torch
from monai import transforms 
from tqdm import tqdm
from monai.data import DataLoader
from torch.utils.data import Dataset
import os


       
class GetPatch(object):
    def __init__(self,patchsize):
        self.patchsize=patchsize
        
    def __call__(self,image,mask=None):
        if mask is None:
            
            return self._patch(image)
        else:
            return self._patch(image),self._patch(mask)
    
    def _patch(self,src):
        
        n=self.patchsize
        
        cnt = 0
        imglist=[]
        h,w,d=src.shape
        
        maxiter=np.max([h,w,d]) // n
#         print(maxiter)
        for i in range(maxiter):
            for j in range(maxiter):
                for k in range(maxiter):
                    
                    imglist.append(src[i*n:(i+1)*n,j*n:(j+1)*n,k*n:(k+1)*n][None,...])
                    
#         print(len(imglist),imglist[0].shape)       
        return torch.cat(imglist,0)


class Dataset(Dataset):
    

    def __init__(self ,mode='train',transformer=False):
        datacsv=pd.read_csv('/data1/dataset/splitdataTrainvalidTest.csv')

        mode2mod={'train':'isTraindata','valid':'isValiddata','test':'isTestdata'}

        self.mode=mode
        
        
        self.KEYS=('image','mask')
        
        
        
        mod=mode2mod[self.mode]
        
        if self.mode=='train':
            
            self.plug_names = datacsv[(datacsv['discard']==False)  & (datacsv[mod]==True)]['filename'].to_list()
            self.plug_label_map = datacsv[(datacsv['discard']==False)   & (datacsv[mod]==True)]['label'].to_list()
            
        else:
            self.plug_names = datacsv[(datacsv['discard']==False)   & (datacsv[mod]==True)]['filename'].to_list()
            self.plug_label_map = datacsv[(datacsv['discard']==False)   & (datacsv[mod]==True)]['label'].to_list()
            
            
  
    
    
        
        self.label_encoder={
                            '0':0,
                            '1':1,
                            '2':1,
                            '3':1,
                            '4':1
                            }
        
        self.transformer=transformer
        
        self.getpatch=GetPatch(32)
        

    def __len__(self):
        return len(self.plug_names)
    

    def _transform(self, data):
        if self.mode!='train' or self.transformer:
            transform=transforms.Compose([
                #     LoadImageD(KEYS),
                    transforms.EnsureChannelFirstD("image"),
                    transforms.AddChannelD("mask"),
                #     OrientationD(KEYS, axcodes='RAS'),
                #     SpacingD(KEYS, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
                    transforms.ScaleIntensityD(keys="image"),
                #     ResizeD(KEYS, (64, 64, 32), mode=('trilinear', 'nearest')),
#                     RandAffineD(KEYS, spatial_size=(-1, -1, -1),
#                                 rotate_range=(0, 0, np.pi/2),
#                                 scale_range=(0.1, 0.1),
#                                 mode=('bilinear', 'nearest'),
#                                 prob=1.0),
                    transforms.NormalizeIntensityD(keys='image'),
                    transforms.ToTensorD(self.KEYS),
                ])
          
        else:
            transform = transforms.Compose([
                #     transforms.LoadImageD(KEYS),
                    transforms.EnsureChannelFirstD("image"),
                    transforms.AddChannelD("mask"),
                
                    transforms.ScaleIntensityD(keys="image"),
                    transforms.OneOf([transforms.RandSpatialCropD(self.KEYS,roi_size=(80,80,80)),
                    
                #     ResizeD(KEYS, (64, 64, 32), mode=('trilinear', 'nearest')),
                    transforms.RandAffineD( self.KEYS,spatial_size=(-1, -1, -1),
                                translate_range=(32, 32, 32),
                                # translate_range=(16, 16, 16),
                                # translate_range=(8, 8, 8),
                                
                                rotate_range=(0, np.pi/4, 0),
                                # rotate_range=(0, np.pi/12, 0),
                                scale_range=(0.1, 0.1,0.1),
                                mode=('bilinear', 'nearest'),
                                prob=1.0)],[0.5,0.5]),
                    transforms.RandAxisFlipD(self.KEYS,prob=0.3),
                    
                    transforms.NormalizeIntensityD(keys='image'),
                    transforms.ResizeD(self.KEYS,(128,128,128), mode=('trilinear', 'nearest')),
                    transforms.ToTensorD(self.KEYS),
                ])
           
        
        
        return transform(data)
    
    def getfiles(self):
        
        return self.plug_names
        
    
    def __getitem__(self, index):

        plug_name = self.plug_names[index]
        
        plug_name=os.path.join('/data1/dataset/ct_box_3d_128_clip_npy/',plug_name.split('/')[-1])
        
        if self.mode=='train':
            maskpath=os.path.join('/data1/dataset/masks',plug_name.split('/')[-1])
        else:
            maskpath=plug_name
            
        dict_loader = transforms.LoadImageD(keys=("image", "mask"))
        data = dict_loader({"image": plug_name, 
                             "mask": maskpath})
        
        data=self._transform(data)
        
        label = self.label_encoder[str(self.plug_label_map[index])]
#         image_array=np.load(plug_name,allow_pickle=True)
#         img=image_array
#         img=self._transform(image_array)
    
#         label = self.label_encoder[str(self.plug_label_map[index])]

        return self.getpatch(data['image'][0])[:,None,...],self.getpatch(data['mask'][0])[:,None,...],torch.tensor(label,dtype=torch.long)







if __name__=='__main__':
    
        
    mytraindataset=Dataset( 'train' )

    myvaliddataset=Dataset( 'valid' )

    mytestdataset=Dataset( 'test' )


    train_loader = DataLoader(mytraindataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = DataLoader(myvaliddataset, batch_size=1, shuffle=False, num_workers=1)
    test_loader = DataLoader(mytestdataset, batch_size=1, shuffle=False, num_workers=1)

    
    
    for i,m,l in mytraindataset:
        # i=torch.reshape(i,(128,128,128))
        print(i.shape,l,m.shape)
        
        # showplt(i,l)
        # showAsrc(i,l,A)
        print(i.min(),i.max())
        break

    for i,_,l in myvaliddataset:
        print(i.shape,l)
        print(i.min(),i.max())
        break

    for i,l in mytestdataset:
        print(i.shape,l)
        print(i.min(),i.max())
        break
     
    print(len(mytraindataset),len(myvaliddataset),len(mytestdataset))
    
    
    
    # for i,m,l in train_loader:
    #     print(i.shape,l,m.shape)
        
    #     # showplt(i,l)
    #     # showAsrc(i,l,A)
    #     print(i.min(),i.max())
    #     break
        
    
    
