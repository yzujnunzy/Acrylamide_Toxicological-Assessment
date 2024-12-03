import os
import math
import argparse
import datetime
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from models.mobileNetv2 import MobileNetV2
from models.mobileNetv3 import MobileNetV3
from models.RepVGG_224px import create_RepVGG_B1
# from models.RepVGG_45px import create_RepVGG_B1
from models.VGG import vgg
from models.GoolgNet import GoogleNet
from models.efficientNet import efficientnet_b0
from models.resnet50_spd import resnet34_spd
from models.resnet import resnet34, resnet50
from models.densenet import densenet121, load_state_dict
from utils.my_dataset import MyDataSet
from utils.utils import read_split_data, train_one_epoch, evaluate
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Instantiating the training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Instantiating a Validation Dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # Load pre-trained weights if present
    # model = DarkNet53(class_dim=args.num_classes).to(device)
    # model = densenet121(num_classes=args.num_classes).to(device)
    model = resnet34(num_class=args.num_classes).to(device)
    # model = resnet34_spd(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            load_state_dict(model, args.weights)
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # Whether to freeze weights
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights are frozen except for the last fully connected layer
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    accuracy = 0

    results_file = ".//process//Resnet34_224-{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_path = 'XXX'

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()
        start_time = time.time()

        all_labels, all_preds = [], []
        model.eval()
        # validate
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                preds = torch.max(outputs, dim=1)[1]
                preds_np = preds.cpu().numpy()
                max_label = preds_np.max()  # Or set the maximum number of categories based on the dataset
                for label in preds_np:
                    all_preds.append(label)
                labels_np = labels.cpu().numpy()
                max_label = labels_np.max()
                for label in labels_np:
                    all_labels.append(label)

        end_time = time.time()
        analyze_time = end_time - start_time

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        if acc >= accuracy:
            accuracy = acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        # Documenting the process
        with open(results_file, 'a') as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\t" \
                         f"correct: {acc:.3f}\t" \
                         f"precision:{precision:.3f}\t" \
                         f"recall:{recall:.3f}\t" \
                         f"f1_score:{f1:.3f}\t" \
                         f"analyze_time:{analyze_time}\n"
            f.write(train_info + "\n\n")
        print('[epoch%d] train_loss:%.3f test_accuracy:%.3f' % (epoch + 1, mean_loss, acc))
        print(f'{precision:.3f},{recall:.3f},{f1:.3f}')
    print(f'best_acc={accuracy},best_epoch={best_epoch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--data_path', type=str,
                        default=r"../data/seg")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
