import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *
from torch.utils.data import ConcatDataset, Subset
from UNet import FAT_Net
from sklearn.model_selection import KFold
from engine import *
import os
import sys
import pandas as pd # <-- Includes pandas for Excel export
import numpy as np  # <-- Needed for averaging
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def main(config):

    # K-FOLD MODIFICATION: Set number of folds
    n_folds = 5
    # K-FOLD MODIFICATION: Store test metrics for each fold
    all_fold_metrics = []

    print('#----------Preparing dataset----------#')
    # K-FOLD MODIFICATION: Load train and val, then combine them for K-fold split
    # We load them once, outside the loop.
    full_train_dataset = isic_loader(path_Data = config.data_path, train = True)
    full_val_dataset = isic_loader(path_Data = config.data_path, train = False)
    combined_dataset = ConcatDataset([full_train_dataset, full_val_dataset])
    
    # K-FOLD MODIFICATION: Load the test set once. It will be used to evaluate each fold's best model.
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True, 
                                 num_workers=config.num_workers,
                                 drop_last=True)

    # K-FOLD MODIFICATION: Initialize KFold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    # K-FOLD MODIFICATION: Start the cross-validation loop
    for fold, (train_indices, val_indices) in enumerate(kfold.split(combined_dataset)):
        print(f'#=============== FOLD {fold + 1}/{n_folds} ===============#')

        # K-FOLD MODIFICATION: Create fold-specific directories
        fold_work_dir = os.path.join(config.work_dir, f'fold_{fold + 1}')
        log_dir = os.path.join(fold_work_dir, 'log')
        checkpoint_dir = os.path.join(fold_work_dir, 'checkpoints')
        resume_model = os.path.join(checkpoint_dir, 'latest.pth')
        outputs = os.path.join(fold_work_dir, 'outputs')
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(outputs):
            os.makedirs(outputs)

        print('#----------Creating logger----------#')
        global logger
        logger = get_logger(f'train_fold_{fold + 1}', log_dir)
        
        log_config_info(config, logger)
        logger.info(f'Starting Fold {fold + 1}/{n_folds}')


        print('#----------GPU init----------#')
        set_seed(config.seed)
        gpu_ids = [0]# [0, 1, 2, 3]
        torch.cuda.empty_cache()


        print('#----------Preparing dataset for Fold {}----------#'.format(fold + 1))
        # K-FOLD MODIFICATION: Create Subsets for train and validation
        train_subset = Subset(combined_dataset, train_indices)
        val_subset = Subset(combined_dataset, val_indices)

        train_loader = DataLoader(train_subset,
                                  batch_size=config.batch_size, 
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=config.num_workers)
        val_loader = DataLoader(val_subset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True, 
                                  num_workers=config.num_workers,
                                  drop_last=True)
        # Note: test_loader is already created outside the loop


        print('#----------Prepareing Models----------#')
        # K-FOLD MODIFICATION: Re-initialize model for each fold
        model_cfg = config.model_config
        model=FAT_Net(
            n_classes=model_cfg['num_classes'], 
            n_channels=model_cfg['input_channels']
        )
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])


        print('#----------Prepareing loss, opt, sch and amp----------#')
        # K-FOLD MODIFICATION: Re-initialize all training components
        criterion = config.criterion
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
        scaler = GradScaler()


        print('#----------Set other params----------#')
        # K-FOLD MODIFICATION: Reset params for each fold
        min_loss = 999
        start_epoch = 1
        min_epoch = 1
        
        # <-- EARLY STOPPING: 1. Initialize counter and patience ---
        # (Make sure 'config.patience' is set in your config file, e.g., 10 or 20)
        early_stopping_counter = 0
        patience = config.patience 
        # --- End of change ---


        if os.path.exists(resume_model):
            print('#----------Resume Model and Other params----------#')
            checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            saved_epoch = checkpoint['epoch']
            start_epoch += saved_epoch
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
            logger.info(log_info)


        print('#----------Training----------#')
        for epoch in range(start_epoch, config.epochs + 1):

            torch.cuda.empty_cache()

            train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )

            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config
                )

            # <-- EARLY STOPPING: 2. Update logic to use counter ---
            if loss < min_loss:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch
                early_stopping_counter = 0 # Reset counter on improvement
                logger.info(f"New best model! Val loss: {min_loss:.4f}. Counter reset.")
            else:
                early_stopping_counter += 1 # Increment counter on no improvement
                logger.info(f"No improvement. Early stopping counter: {early_stopping_counter}/{patience}")
            # --- End of change ---


            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')) 

            
            # <-- EARLY STOPPING: 3. Add check to break loop ---
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
                break # Exit the epoch loop for this fold
            # --- End of change ---


        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            # K-FOLD MODIFICATION: Load best model for this fold
            best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
            model.module.load_state_dict(best_weight)
            
            # K-FOLD MODIFICATION: Run test and get metrics dictionary
            test_metrics = test_one_epoch(
                    test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    outputs,
                    test_data_name=f'fold_{fold + 1}' 
                )
            # K-FOLD MODIFICATION: Store metrics for this fold
            all_fold_metrics.append(test_metrics)
            
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            ) 
        
        print(f'#=============== FOLD {fold + 1} COMPLETE ===============#')
        torch.cuda.empty_cache() # Clean up GPU memory before next fold

    # K-FOLD MODIFICATION: After all folds are done, calculate and log average metrics
    print('#----------All Folds Complete - Averaging Results----------#')
    if all_fold_metrics:
        # Calculate mean and std deviation for each metric
        avg_metrics = {}
        std_metrics = {}
        metric_keys = all_fold_metrics[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in all_fold_metrics]
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
            
        # --- NEW CODE TO SAVE TO EXCEL ---
        print('#----------Saving results to Excel----------#')
        
        # 1. Create a DataFrame from the list of fold metrics
        df_folds = pd.DataFrame(all_fold_metrics)
        # Set the index to be 'Fold 1', 'Fold 2', etc.
        df_folds.index = [f'Fold {i+1}' for i in range(len(all_fold_metrics))]
        
        # 2. Create DataFrames for the 'Average' and 'StdDev'
        df_avg = pd.DataFrame(avg_metrics, index=['Average'])
        df_std = pd.DataFrame(std_metrics, index=['StdDev'])

        # 3. Combine the fold results, average, and std dev into one table
        df_final_results = pd.concat([df_folds, df_avg, df_std])
        
        # 4. Define the save path (e.g., in your main working directory)
        excel_path = os.path.join(config.work_dir, 'k_fold_test_results.xlsx')
        
        # 5. Save the DataFrame to an Excel file
        df_final_results.to_excel(excel_path)
        
        print(f"Successfully saved all fold results to {excel_path}")
        # --- END OF NEW CODE ---
        

        # Log the final averaged results (Your existing code)
        final_log_dir = os.path.join(config.work_dir, 'log') # A central log
        final_logger = get_logger('final_results', final_log_dir)
        
        final_logger.info('=' * 30)
        final_logger.info(f'{n_folds}-FOLD CROSS-VALIDATION FINAL RESULTS')
        final_logger.info('=' * 30)
        
        for key in metric_keys:
            final_logger.info(f'Average {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}')
        
        print(f"Final averaged results logged to {final_log_dir}")
        for key in metric_keys:
            print(f'Average {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}')

    else:
        print("No test metrics were collected.")


if __name__ == '__main__':
    config = setting_config
    main(config)