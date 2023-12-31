import os
import argparse
import torch

import torch.optim as optim
import albumentations as A

from albumentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2


from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms

from my_dataset import MyDataSet
# from model_v2 import swinv2_tiny_patch4_window8_256 as create_model
from model_v1 import swin_base_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform ={
        'train': Compose([A.RandomRotate90(),
                               A.Flip(),
                               OneOf([
                                   transforms.HueSaturationValue(),
                                   transforms.RandomBrightness(),
                                   transforms.RandomContrast(),
                                   transforms.RandomBrightnessContrast(),
                                   ], p=1),
                                   A.Resize(args.img_size,args.img_size),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                   ToTensorV2(),
                                   ]),

        'val': Compose([A.Resize(args.img_size,args.img_size),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             ToTensorV2(),
                             ])}
    
    # Training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)


    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # Delete the weight of category in classification
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights are frozen except for head
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9,0.9999), eps=1e-08)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/modelV1-{}_CTbase32.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-05)

    # Root directory of the train dataset
    parser.add_argument('--data-path', type=str,
                        default="dataset/CT/train")

    # Pretrain weight path, set to null character if you do not to load the Pretrain weight
    parser.add_argument('--weights', type=str, 
                        default='pretrained_model/swin_base_patch4_window7_224.pth',
                        help='initial weights path')
    
    # Whether to freeze the weights
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
 
    main(opt)


