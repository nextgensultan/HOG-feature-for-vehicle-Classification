from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn

class HOG:
    def __init__(self,bins,PixelsPerCell,CellsPerBlock):
        self.binCount = bins
        self.PixelsPerCell = PixelsPerCell
        self.CellsPerBlock  = CellsPerBlock   
        # self.count = 0
    def GetFeatures(self,img):
        # print(img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])
        img = transform(img)
        BinArr = self.CreateBins(img)
        # self.count+=1
        return self.MakeFeatures(BinArr)
        
    def MakeFeatures(self,BinsArr):
        Features = list()
        for i in range(0,BinsArr.shape[0]-self.CellsPerBlock+1):
            for j in range(0,BinsArr.shape[1] - self.CellsPerBlock+1):
                blockFeature = np.ravel(BinsArr[i:i+self.CellsPerBlock, j:j+self.CellsPerBlock],order = 'F')
                # lets normalize the data
                epsilon = 1e-10
                blockFeature /= np.sqrt(np.sum(np.square(blockFeature)) + epsilon)
                Features.append(blockFeature)
        return np.ravel(np.array(Features))
        
    def ConvertArrayToBins(self,cell):
        """
        Cell is the current Cell we are working on and it retrns the calculated
        Histogram that will be used later as a feature
        """
        padded = np.pad(cell, 1, mode='constant', constant_values=0)
        Gx = padded[1:-1, 2:] - padded[1:-1, :-2]+0.0001
        Gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        mag = np.sqrt(Gx**2 + Gy**2)
        theta = np.round(np.degrees(np.arctan2(Gy, Gx))) % 180
        
        bin_size = 180 / self.binCount
        BinIdx = (theta / bin_size).astype(int)
        # print(theta[BinIdx == 9])
        # print("idx" , self.count)
        lower = BinIdx * bin_size
        upper = lower + bin_size
        contribution = (upper - theta) / bin_size * mag
    
        Hist = np.zeros(self.binCount)
        np.add.at(Hist, BinIdx, contribution)
        np.add.at(Hist, (BinIdx + 1) % self.binCount, mag - contribution)
        return Hist

        
    def CreateBins(self,img):
        inc = 180 / self.binCount
        histArr = list()
        for i in range(0,img.shape[1],self.PixelsPerCell):
            HistRow = list()
            for j in range(0,img.shape[2],self.PixelsPerCell):
                Cell = img[0,i:i+self.PixelsPerCell , j:j+self.PixelsPerCell]
                HistRow.append(self.ConvertArrayToBins(Cell))
            histArr.append(np.array(HistRow))
        return np.array(histArr)