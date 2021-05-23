import argparse
import numpy as np


import torch

from torchvision import transforms

import torch.backends.cudnn as cudnn

import dataset
import multi_channel_resnet34_hyper

from scipy import stats
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="No reference 360 degree image quality assessment.")
    parser.add_argument('--gpu', dest='gpu_id', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
          default='CVIQ', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_test', dest='filename_test', help='Test csv file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    gpu = args.gpu_id
    snapshot = args.snapshot
    database = args.database

    filename_test = args.filename_test


    torch.cuda.set_device(gpu)


    
    # load the network
    model = multi_channel_resnet34_hyper.resnet34(pretrained = False)
    save_sata_dict = torch.load(snapshot)
    model.load_state_dict(save_sata_dict)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),\
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    test_dataset = dataset.Dataset(args.data_dir, filename_test, transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model.cuda()
   
    with torch.no_grad():
        model.eval()
        label=np.zeros([len(test_dataset)])
        y_output=np.zeros([len(test_dataset)])
        for i, (image_BA, image_BO, image_F, image_L, image_R, image_T, mos) in enumerate(test_loader):

            image_BA = image_BA.cuda()
            image_BO = image_BO.cuda()
            image_F = image_F.cuda()
            image_L = image_L.cuda()
            image_R = image_R.cuda()
            image_T = image_T.cuda()

            mos = mos.cuda()

            label[i] = mos.item()

            mos_predict = model(image_BA,image_BO,image_F,image_L,image_R,image_T)

            y_output[i] = mos_predict.item()
        

        label = np.array(label)
        label = label.reshape(int(len(test_dataset)/180), 180)
        label = np.mean(label, axis=1)

        y_output = np.array(y_output)
        y_output = y_output.reshape(int(len(test_dataset)/180), 180)
        y_output = np.mean(y_output, axis=1)

        y_output_logistic = fit_function(label, y_output)
        val_PLCC = stats.pearsonr(y_output_logistic, label)[0]
        val_SRCC = stats.spearmanr(y_output, label)[0]
        val_KRCC = stats.stats.kendalltau(y_output, label)[0]
        val_RMSE = np.sqrt(((y_output_logistic-label) ** 2).mean())
        
        print('Test completed.')
        print('SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(\
            val_SRCC, val_KRCC, val_PLCC, val_RMSE))