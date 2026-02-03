import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils2 import save_imgs





def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

    

        if config.amp:
            with autocast():
                out = model(images) 
                
                # --- DEBUG CHECK 2: Model Logits (AMP) ---
                if torch.isnan(out).any():
                    raise RuntimeError(f"!!! ERROR: NaN detected in model output (logits) at epoch {epoch}, iter {iter} (AMP) !!!")

                loss = criterion(out, targets.float()) 
            
          
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            scaler.step(optimizer)
            scaler.update()

        else:
            out = model(images)

            loss = criterion(out, targets.float())
            
          

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()

def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)           # 1. Get logits
          
            loss = criterion(out.float(), msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)




def test_one_epoch(test_loader,
                     model,
                     criterion,
                     logger,
                     config,
                     outputs=None,           # <-- FIX 1: Add 'outputs' parameter
                     test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            
            out = model(img)
            
            loss = criterion(out.float(), msk)
            loss_list.append(loss.item())
            
            # Prepare ground truth
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            
            # Prepare prediction
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            
            # <-- FIX 2: Use the 'outputs' variable, not config.work_dir
            if outputs is not None:
                save_imgs(img, msk, out, i, outputs, config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        
        # Handle cases where confusion matrix might be incomplete (e.g., only one class predicted)
        if confusion.shape == (1, 1):
            # Only one class present and predicted correctly or incorrectly
            if y_true[0] == 1: # Only positives
                TP = confusion[0,0]
                FN = 0
                TN = 0
                FP = 0
            else: # Only negatives
                TP = 0
                FN = 0
                TN = confusion[0,0]
                FP = 0
        elif confusion.shape == (2, 2):
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
        else:
            # Fallback for unexpected confusion matrix shapes
            logger.warning(f"Unexpected confusion matrix shape: {confusion.shape}. Metrics will be 0.")
            TN, FP, FN, TP = 0, 0, 0, 0


        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
            
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                     specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

        # <-- FIX 3: Return a dictionary of all metrics for the Excel file
        metrics_dict = {
            'Test_Loss': np.mean(loss_list),
            'mIoU': miou,
            'Dice': f1_or_dsc,
            'Accuracy': accuracy,
            'Specificity': specificity,
            'Sensitivity': sensitivity
        }
        
        return metrics_dict