import os
import argparse

import torch

import albumentations as A

from albumentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from my_dataset import MyDataSet
# from model_v2 import swinv2_tiny_patch4_window8_256 as create_model
from model_v1 import swin_small_patch4_window7_224
from utils import read_test_data, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if os.path.exists('./weights') is False:
        os.mkdir('./weights')

    # writer = SummaryWriter()

    test_images_path, test_images_label = read_test_data(args.data_path)

    data_transform = Compose([A.Resize(args.img_size, args.img_size),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                              ToTensorV2(),])
    
    # testing dataset
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform) 
    
    nw = min([os.cpu_count(), args.batch_size if args.batch_size>1 else 0, 8])    #number of workers 
    print('Using {} dataloader workers every process'.format(nw))
    # load test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)
    
    # create model
    model = swin_small_patch4_window7_224(num_classes=args.num_classes, img_size=args.img_size).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    for epoch in range(args.epochs):
        test_loss, test_acc = evaluate(model=model,
                                       data_loader=test_loader,
                                       device=device,
                                       epoch=epoch)

    tags = ['test_loss', 'test_acc']
    # writer.add_scalar(tags[0], test_loss, epoch)
    # writer.add_scalar(tags[1], test_acc, epoch)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=str, default=4)
    parser.add_argument('--img_size', type=str, default=224)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)

    # Root directory of the test dataset
    parser.add_argument('--data-path', type=str,
                        default='dataset/CT/test')
    
    parser.add_argument('--weights', type=str, 
                        default='weights/model-7_CTsma32.pth',
                        help='weights path')
    
    opt = parser.parse_args()

    main(opt)
        