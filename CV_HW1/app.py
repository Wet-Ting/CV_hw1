############################################################################
#                              Cv.hw1                                      #  
#                        Arthor: Wet-ting Cao.                             #   
#                             2019.10.24                                   #
############################################################################

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsView, QGraphicsScene
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from cv import Ui_MainWindow
import cv2 as cv
import numpy as np  

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math

global pt
pt = []

# MainWindow -> button implementation.
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        # Q1.
        self.Findcorners.clicked.connect(self.findCorner)
        self.Intrinsic.clicked.connect(self.intrinsic)
        
        # Q2.
        self.AR.clicked.connect(self.ar)
        
        # Q3.
        self.RST.clicked.connect(self.rst)
        self.PT.clicked.connect(self.pt)
        
        # Q4.
        self.FindContour.clicked.connect(self.findContour)

        # Q5.
        self.Imgs.clicked.connect(self.showImgs)
        self.Hyper.clicked.connect(self.showHyper)
        self.Train.clicked.connect(self.train)
        self.Result.clicked.connect(self.showResult)
        self.Inf.clicked.connect(self.showInf)
        
    # 1.1 Find corners
    def findCorner(self) :
        path = 'images/images/CameraCalibration/'
        for i in range(15) :
            img = cv.imread(path + str(i + 1) + '.bmp')
            to_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corner = cv.findChessboardCorners(to_gray, (11, 8), None)
            
            if ret == True :               
                img = cv.drawChessboardCorners(img, (11, 8), corner, ret)
                cv.namedWindow('findCorners', cv.WINDOW_NORMAL)
                cv.resizeWindow('findCorners', 1024, 1024)
                cv.imshow('findCorners', img)
                cv.waitKey(1000)
                        
        cv.destroyAllWindows()
    
    # 1.2 Find the Intrinsic Matrix
    def intrinsic(self) :
        pass
        
        
    # 2. Augmented Reality
    def ar(self) :
        print('Augmented Reality')
        def draw(img, corner, imgpt) :
            imgpt = np.int32(imgpt).reshape(-1, 2)
            img = cv.drawContours(img, [imgpt[:4]], -1, (0, 0, 255), 3)
            for i in range(4) :
                img = cv.line(img, tuple(imgpt[i]), tuple(imgpt[i + 4]), (0, 0, 255), 3)
                
            return img
            
        images = []
        objp = []
        imgp = []
        path = 'images/images/CameraCalibration/'
        obj = np.zeros((11 * 8, 3), np.float32)
        obj[:, : 2] = np.mgrid[0 : 11, 0 : 8].T.reshape(-1, 2)
        criter = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        axis = np.float32([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0],
                   [0, 0, -2], [0, 0, -2], [0, 0, -2], [0, 0, -2]])
        
        for i in range(5) :
            img = cv.imread(path + str(i + 1) + '.bmp')
            images.append(img)
        
        for img in images:
            to_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corner = cv.findChessboardCorners(to_gray, (11, 8), None)
            
            if ret == True :
                objp.append(obj)
                corners = cv.cornerSubPix(to_gray, corner, (11, 11), (-1, -1), criter)
                imgp.append(corners)
                
                img = cv.drawChessboardCorners(img, (11, 8), corner, ret)
        
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', 512, 512)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objp, imgp, to_gray.shape[::-1], None, None)
        
        images.clear()
        for i in range(5) :
            img = cv.imread(path + str(i + 1) + '.bmp')
            images.append(img)
        
        for img in images:
            to_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corner = cv.findChessboardCorners(to_gray, (11, 8), None)
            if ret == True :
                corners = cv.cornerSubPix(to_gray, corner, (11, 11), (-1, -1), criter)
                _, rvecs, tvecs, inliers = cv.solvePnPRansac(obj, corners, mtx, dist)
                imgpt, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = draw(img, corners, imgpt)
                cv.imshow('Image', img)
                cv.waitKey(500)
                
        cv.destroyAllWindows()
    
    # 3.1 Transforms: Rotation, Scaling, Translation
    def rst(self) :
        print('Transforms: Rotation, Scaling, Translation')       
        def translate(img, x, y):
            M = np.float32([[1, 0, x], [0, 1, y]])
            shifted = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
            return shifted
            
        def rotate(img, angle, scale = 1.0):
            ret, thresh = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 127, 255, 0)
            _, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            M = cv.moments(contours[0])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            h, w = img.shape[:2]
            center = (cx - 2, cy - 3)
                       
            M = cv.getRotationMatrix2D(center, angle, scale)
            rotated = cv.warpAffine(img, M, (w, h))
            
            return rotated
            
        img = cv.imread('images/images/OriginalTransform.png')
        angle = self.angle.toPlainText()
        scale = self.scale.toPlainText()
        tx = self.tx.toPlainText()
        ty = self.ty.toPlainText()
        
        translated = translate(img, float(tx), float(ty))
        rotated = rotate(translated, float(angle), float(scale))
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')
        plt.subplot(122), plt.axis('off'), plt.imshow(cv.cvtColor(rotated, cv.COLOR_BGR2RGB)), plt.title('Tansforms')
        plt.show()  

    # 3.2 Perspective Transformation
    def pt(self) :
        global x0, y0, pt
        pt = []
        print('Perspective transform')
        def getPoint(event, x, y, flags, param):
            global x0, y0, pt
            if event == cv.EVENT_LBUTTONDOWN:
                x0, y0 = x, y
                pt.append([x0, y0])
           
        img = cv.imread('images/images/OriginalPerspective.png')
        cv.namedWindow('img')
        cv.setMouseCallback('img', getPoint)
        while (1):
            k = cv.waitKey(33)
            cv.imshow('img', img)
            if len(pt) == 4: 
                break
                
        print('pt1: (' + str(pt[0][0]) + ', ' + str(pt[0][1]) + ')')
        print('pt2: (' + str(pt[1][0]) + ', ' + str(pt[1][1]) + ')')
        print('pt3: (' + str(pt[2][0]) + ', ' + str(pt[2][1]) + ')')      
        print('pt4: (' + str(pt[3][0]) + ', ' + str(pt[3][1]) + ')')
        
        cv.destroyWindow('img')
        
        pts1 = np.float32([pt[0], pt[1], pt[3], pt[2]])
        pts2 = np.float32([[20, 20], [450, 20], [20, 450], [450,450]])
        
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(img, matrix, (450, 450))
        
        cv.imshow('Result', result)
        print('Enter any keys to close this window.')
        cv.waitKey(0)
        cv.destroyWindow('Result')
                           
    # 4 Find contours.
    def findContour(self) :
        print('Fine contours')
        image = cv.imread('images/images/Contour.png')
        
        to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)        
        edged = cv.Canny(to_gray, 30, 200)
        _, contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        cont = image.copy()
        cv.drawContours(cont, contours, -1, (0, 0, 255), 2)
        result = np.hstack([image, cont])
        cv.imshow('Original & Result', result)
        
    # 5.1 show training imgs.
    def showImgs(self) :
    
        def imshow(img, label, i):         
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()              
            plt.subplot(1, 10, i + 1)
            plt.title(classes[label])
            plt.axis('off')
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            
        print('Show training images.')
        # get some random training images
        dataiter = iter(show)
        images, labels = dataiter.next()

        # show images
        for i in range(10):
            imshow(torchvision.utils.make_grid(images[i]), labels[i], i)
        plt.show()
        
    # 5.2 show Hyperparams.
    def showHyper(self) :
        print('Show hyperparameters!')
        print('batch size: ' + str(batch))
        print('learning rate: ' + str(lr))
        print('optimizer: ' + optimizer)
        
    # 5.3 train 1 epoch.
    def train(self) :
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Lenet5().to(device)
        criteon = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr = lr)
        loss_iter = []
        iter = 0
        
        print('Start to train 1 epoch...')
        for epoch in range(1):
            model.train()
            for i, (x, label) in enumerate(trainloader) :
                x, label = x.to(device), label.to(device)
                y = model(x)
                loss = criteon(y, label)
                loss_iter.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter += 1
                
            plt.figure()
            plt.plot(loss_iter)
            plt.title('model loss (1 epoch): ' + str((sum(loss_iter).item() / iter)))
            plt.ylabel('loss'), plt.xlabel('iteration')
            plt.legend(['training loss'], loc = 'upper left')
            plt.show()
            
        print('Training loss at the end of the epoch is: ' + str((sum(loss_iter).item() / iter)))
    
    # 5.4 Show training result.
    def showResult(self) :
        acc = mpimg.imread('acc.png')
        loss = mpimg.imread('loss.png')
        
        plt.figure(figsize = (10, 12))
        plt.subplot(2, 1, 1)
        plt.axis('off')
        plt.imshow(acc)
        plt.subplot(2, 1, 2)
        plt.axis('off')
        plt.imshow(loss)
        plt.show()
        
    # 5.5 Inference
    def showInf(self) :    
        def imshow(img):         
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.figure(figsize = (15, 10))
            plt.subplot(1, 2, 1)
            plt.title('Show image')
            plt.axis('off')
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Lenet5().to(device)
        pretrained = torch.load('cifar_lenet5.pth')
        model.load_state_dict(pretrained)
        model.eval()
        num = self.Test_idx.toPlainText()
        print('You choose: ' + num)
        num = int(num)
        
        group = math.floor(num / batch)
        num %= batch
                
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        for _ in range(group):
            images, labels = dataiter.next()
        
        imshow(torchvision.utils.make_grid(images[num]))
        
        with torch.no_grad():
            prob = []
            x, label = images[num].to(device), labels[num].to(device)
            x = x.unsqueeze(0)
            y = model(x)
            y = F.softmax(y, dim = 1).cpu()
            for i in range(10):
                prob.append(y[0][i].item())
            plt.subplot(1, 2, 2)
            plt.bar(classes, prob)
            plt.title('Estimation result')
            plt.ylabel('prob'), plt.xlabel('classes')
            plt.show()
                    
############################################################################
#                                                                          #
#              Q5. train a Cifar-10 classifier using Lenet-5               #        
#                                                                          #
############################################################################

batch = 128
lr = 0.01
optimizer = 'SGD'
    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

show = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
show = torch.utils.data.DataLoader(show, batch_size=10,
                                          shuffle=True, num_workers=2)  
                                         
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)                                          

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        
        # (3, 32, 32) -> (6, 14, 14) -> (16, 5, 5)
        self.conv = nn.Sequential( 
            nn.Conv2d(3, 6, 5, 1, 0), # in_ch, out_ch, kernel, stride, padding.
            nn.AvgPool2d(2, 2, 0),  # kernel, stride, padding.
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.AvgPool2d(2, 2, 0)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(), 
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv(x)        
        x = x.view(batchsize, 16 * 5 * 5)
        x = self.fc(x) # logits
        return x

############################################################################
#                                                                          #
#        end   Q5. train a Cifar-10 classifier using Lenet-5               #        
#                                                                          #
############################################################################

                              
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
