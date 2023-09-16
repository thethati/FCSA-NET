import torch
import pytz
import logging
import os

from myconfig import args_info, device
from mymodel import fmodel

from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_auc_score
from utils import DataLoader, BenchmarkDataset
from utils.tools import model_size, model_save
from datetime import datetime
from tqdm import *



def train(model, optimizer, train_loader, epoch):
    model.train()
    for i, train_batch in enumerate(train_loader):
        train_batch = train_batch.to(device)
        ret = model(train_batch)
        bpr_loss , batch_acc = model.bpr_loss(ret)

        #img_loss = 5e-3 * img_loss

        loss = bpr_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not (i % 100):
            logging.debug(f"Epoch[{epoch + 1}][{i + 1}/{len(train_loader)}] "
                          f"tot_loss: {loss.item():.5f} "
                          f"bpr_loss: {bpr_loss.item():.5f} "
                          f"batch_acc: {batch_acc:.2f} ")

# inference 
def inference(model, auc_loader, fitb_loader):
    model.eval()
    with torch.no_grad():
        scores, labels = [], []
        for auc_batch in auc_loader:
            auc_batch = auc_batch.to(device)
            auc_score = model.test_auc(auc_batch)
            scores.append(auc_score.cpu())
            labels.append(auc_batch.y.cpu())
          
        scores = torch.cat(scores).numpy()
        labels = torch.cat(labels).numpy()
        cp_auc = roc_auc_score(labels, scores)

        fitb_right = 0
        for fitb_batch in fitb_loader:
            fitb_batch = fitb_batch.to(device)
            fitb_right += model.test_fitb(fitb_batch)
        fitb_acc = fitb_right / len(fitb_loader.dataset)

    logging.debug(f"cp_auc:{cp_auc}%   fitb_acc:{fitb_acc}%")
    

    kpi_total = cp_auc + fitb_acc
    return cp_auc, fitb_acc, kpi_total

def main(args):
    # step 1 : dataloader
    BenchmarkDataset.init(args)
    #pin_memory
    loader_kwargs = {
        'batch_size': args.batch_size, 
        'num_workers': args.num_worker,
        'pin_memory': True
        #'pin_memory': True if torch.cuda.is_available() else False
    }
    # exit()

    train_dataset = BenchmarkDataset('train').next_train_epoch()

    auc_dataset_valid = BenchmarkDataset('valid').test_auc()
    fitb_dataset_valid = BenchmarkDataset('valid').test_fitb()
    auc_loader_valid = DataLoader(auc_dataset_valid, **loader_kwargs)
    fitb_loader_valid = DataLoader(fitb_dataset_valid, **loader_kwargs)


    # step 2 : some configs
    epoches = 60 # or 80
    now_time = datetime.now(tz=pytz.timezone('Asia/Shanghai'))
    print(f'today is : {now_time}')
    remark = now_time.strftime('%Y-%m-%d_%H:%M:%S')
    dirname = 'logs/'
    filename = remark
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] : %(message)s',
                        level= logging.DEBUG,
                        filename= dirname+filename,
                        filemode='a')

    logging.debug("torch.cat([batch_data_attention,batch_data], dim = 0) new gnn  + cat(1 trans,graph-att)  yes resblock for gnn with edgeweight 真正的试炼")

    # step 3 : define model
    model = fmodel(num_node_features=args.hid).to(device)

    # step 4 : define optimize and adjust lr
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    #optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9)
    ajlr = ExponentialLR(optimizer, gamma=1 - 0.015)

    # model size
    print(f'  + Size of params: {model_size(model):.2f}MB')

    # step 5 : train & inference
    best_auc, best_fitb, best_total = 0., 0., 0.
    for epoch in tqdm(range(epoches)):

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        # train for one epoch
        train(model, optimizer, train_loader, epoch)
        # evaluate on validation set
        cp_auc, fitb_acc, kpi_total = inference(model, auc_loader_valid, fitb_loader_valid)
        # update learning rate
        ajlr.step()

        # remember best acc and save checkpoint
        is_best = kpi_total > best_total
        best_auc = max(best_auc, cp_auc)
        best_fitb = max(best_fitb, fitb_acc)
        best_total = max(best_total, kpi_total)

        best_path = model_save(remark, model, epoch, is_best, best_auc=cp_auc, best_fitb=fitb_acc)

        logging.debug(f"best_auc:{best_auc}%   best_fitb:{best_fitb}%   best_total:{best_total}%")
        
        train_dataset.next_train_epoch()


if __name__ == '__main__':
    arg = args_info()
    main(arg)
