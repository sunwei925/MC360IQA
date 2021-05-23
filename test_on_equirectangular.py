import argparse
import numpy as np
import cv2
from PIL import Image

import torch

from torchvision import transforms
import torch.backends.cudnn as cudnn

import eq2cm
import multi_channel_resnet34_hyper


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="No reference 360 degree image quality assessment.")
    parser.add_argument('--gpu', dest='gpu_id', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--filename', dest='filename', help='Test image file.',
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

    filename = args.filename


    torch.cuda.set_device(gpu)


    
    # load the network
    model = multi_channel_resnet34_hyper.resnet34(pretrained = False)
    save_sata_dict = torch.load(snapshot)
    model.load_state_dict(save_sata_dict)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda()

    eqimg = cv2.imread(filename)


    image_BA = eq2cm.eq_to_pers(eqimg, np.pi/2,np.pi, 0, 480, 480)
    image_BO = eq2cm.eq_to_pers(eqimg, np.pi/2,0, np.pi/2, 480, 480)
    image_F = eq2cm.eq_to_pers(eqimg, np.pi/2, 0, 0, 480, 480)
    image_L = eq2cm.eq_to_pers(eqimg, np.pi/2,-np.pi/2, 0, 480, 480)
    image_R = eq2cm.eq_to_pers(eqimg, np.pi/2,np.pi/2, 0, 480, 480)
    image_T =  eq2cm.eq_to_pers(eqimg, np.pi/2,0, -np.pi/2, 480, 480)


    image_BA = Image.fromarray(cv2.cvtColor(image_BA,cv2.COLOR_BGR2RGB))
    image_BO = Image.fromarray(cv2.cvtColor(image_BO,cv2.COLOR_BGR2RGB))
    image_F = Image.fromarray(cv2.cvtColor(image_F,cv2.COLOR_BGR2RGB))
    image_L = Image.fromarray(cv2.cvtColor(image_L,cv2.COLOR_BGR2RGB))
    image_R = Image.fromarray(cv2.cvtColor(image_R,cv2.COLOR_BGR2RGB))
    image_T = Image.fromarray(cv2.cvtColor(image_T,cv2.COLOR_BGR2RGB))

    image_BA = transformations(image_BA)
    image_BO = transformations(image_BO)
    image_F = transformations(image_F)
    image_L = transformations(image_L)
    image_R = transformations(image_R)
    image_T = transformations(image_T)


    # do validation after each epoch
    with torch.no_grad():
        model.eval()

        image_BA = image_BA.cuda().unsqueeze(dim=0)
        image_BO = image_BO.cuda().unsqueeze(dim=0)
        image_F = image_F.cuda().unsqueeze(dim=0)
        image_L = image_L.cuda().unsqueeze(dim=0)
        image_R = image_R.cuda().unsqueeze(dim=0)
        image_T = image_T.cuda().unsqueeze(dim=0)


        quality_score = model(image_BA, image_BO, image_F,image_L, image_R, image_T)

        quality_score = quality_score.item()

        print('The quality score of {} is {:.4f}'.format(filename, quality_score))
        
