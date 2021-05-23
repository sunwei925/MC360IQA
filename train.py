import os, argparse, time

import numpy as np


import torch
import torch.nn as nn

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
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate.',
          default=0.0001, type=float)
    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
          default='CVIQ', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_train', dest='filename_train', help='Training csv file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--filename_test', dest='filename_test', help='Test csv file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--cross_validation_index', dest='cross_validation_index', help='The index of cross validation.',
          default='', type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    snapshot = args.snapshot
    database = args.database
    filename_train = args.filename_train
    filename_test = args.filename_test
    lr = args.lr
    cross_validation_index = args.cross_validation_index

    torch.cuda.set_device(gpu)

    if not os.path.exists(os.path.join(snapshot, database, str(cross_validation_index))):
        os.makedirs(os.path.join(snapshot, database, str(cross_validation_index)))

    
    # load the network
    model = multi_channel_resnet34_hyper.resnet34(pretrained = True)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),\
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = dataset.Dataset(args.data_dir, filename_train, transformations)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = dataset.Dataset(args.data_dir, filename_test, transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model.cuda()

    criterion = nn.MSELoss().cuda()

    # regression loss coefficient
    optimizer = torch.optim.RMSprop(model.parameters(),lr=lr,alpha=0.9)


    print("Ready to train network")

    best_val_criterion = -1  # SROCC min
    best_val = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()   
        session_start_time = time.time()      
        batch_losses = []
        batch_losses_each_disp = []
        for i, (image_BA, image_BO, image_F, image_L, image_R, image_T, mos) in enumerate(train_loader):
            image_BA = image_BA.cuda()
            image_BO = image_BO.cuda()
            image_F = image_F.cuda()
            image_L = image_L.cuda()
            image_R = image_R.cuda()
            image_T = image_T.cuda()
            mos = mos[:,np.newaxis]


            mos = mos.cuda()

            # Forward pass
            mos_predict = model(image_BA,image_BO,image_F,image_L,image_R,image_T)

            # MSE loss
            loss = criterion(mos_predict,mos)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            torch.autograd.backward(loss)
            optimizer.step()

            if (i+1) % 100 == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / 100
                print('Epoch [%d/%d], Iter [%d/%d] Losses: %.4f CostTime: %.4f'
                %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, avg_loss_epoch, session_end_time - session_start_time))    
                session_start_time = time.time()   
                batch_losses_each_disp = []

        avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))


        # do validation after each epoch
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
            
            print('Epoch {} completed. SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                val_SRCC, val_KRCC, val_PLCC, val_RMSE))

            if val_SRCC > best_val_criterion:
                print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                best_val_criterion = val_SRCC
                best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]
                print('Saving model...')

                torch.save(model.state_dict(), os.path.join(snapshot, database, str(cross_validation_index), database + '_epoch_'+ str(epoch+1) + '.pkl'))       
                

    print('Training completed.')
    print('The best training result SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_val[0], best_val[1], best_val[2], best_val[3]))

                






    
