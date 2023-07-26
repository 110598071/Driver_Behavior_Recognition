import pytorch_util, pytorch_model, pytorch_progress
import torch
import time
import optuna
import random

TRIALS = 20
ORIGINAL_EPOCHS = 50
TRANSFER_EPOCHS = [10, 30]
# PYTORCH_SEED = 597381152 ## SLP/MLP
PYTORCH_SEED = 3578671768 ## CNN

def SLP_MLP_hyperparameters_objective(trial):
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 16, 32, 64, 128])

    scheduler_kwargs = {
        'max_lr': trial.suggest_float("max_lr", 0.00005, 0.005, step=0.00005),
        'pct_start': trial.suggest_float("pct_start", 0.1, 0.9, step=0.02),
        'div_factor': trial.suggest_float("div_factor", 5.0, 70.0, step=0.5),
        'final_div_factor': trial.suggest_float("final_div_factor", 0.05, 50.0, step=0.05),
    }
    scheduler_kwargs['total_steps'] = ORIGINAL_EPOCHS

    Adam_kwargs = {
        'lr': scheduler_kwargs['max_lr'],
        'weight_decay': trial.suggest_float("WEIGHT_DECAY", 0.05, 0.5, step=0.005),
    }

    pytorch_util.setup_seed(PYTORCH_SEED)
    model = pytorch_model.get_SLP()
    # model = pytorch_model.get_MLP()

    optimizer = torch.optim.AdamW(model.parameters(), **Adam_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)
    test_accuracy = train_SLP_MLP_and_get_accuracy(optimizer, lr_scheduler, model, BATCH_SIZE, ORIGINAL_EPOCHS)
    return test_accuracy

def train_SLP_MLP_and_get_accuracy(optimizer, lr_scheduler, model, BATCH_SIZE, ORIGINAL_EPOCHS):
    origin_pytorch_model = pytorch_progress.pytorch_model(model, BATCH_SIZE, True)
    origin_pytorch_model.train(optimizer, lr_scheduler, True, ORIGINAL_EPOCHS)
    origin_pytorch_model.test()
    return origin_pytorch_model.get_test_accuracy()

def SLP_MLP_hyperparameter_optimization():
    since = time.time()
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
    study.optimize(SLP_MLP_hyperparameters_objective, n_trials = TRIALS, n_jobs=1)
    time_elapsed = time.time() - since
    print(f'Test model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print('='*30)
    print("hyperparameters optimization result:")
    print()
    print(f"{TRIALS} trials optimization time in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print('Best trial hyperparameters:', study.best_trial.params)
    print("test accuracy: ", study.best_trial.value)
    print('='*30)

def CNN_hyperparameters_objective(trial):
    pytorch_util.setup_seed(PYTORCH_SEED)
    resnet = pytorch_model.get_pretrained_ResNet('152')

    # BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 16, 32])
    BATCH_SIZE = 8

    transfer_fc_scheduler_kwargs = {
        'max_lr': trial.suggest_float("fc_max_lr", 0.00005, 0.005, step=0.00005),
        'pct_start': trial.suggest_float("fc_pct_start", 0.1, 0.9, step=0.02),
        'div_factor': trial.suggest_float("fc_div_factor", 5.0, 70.0, step=1.0),
        'final_div_factor': trial.suggest_float("fc_final_div_factor", 0.05, 50.0, step=0.05),
    }
    transfer_fc_scheduler_kwargs['total_steps'] = TRANSFER_EPOCHS[0]

    transfer_whole_scheduler_kwargs = {
        'max_lr': trial.suggest_float("whole_max_lr", 0.00005, 0.005, step=0.00005),
        'pct_start': trial.suggest_float("whole_pct_start", 0.1, 0.9, step=0.02),
        'div_factor': trial.suggest_float("whole_div_factor", 5.0, 70.0, step=1.0),
        'final_div_factor': trial.suggest_float("whole_final_div_factor", 0.05, 50.0, step=0.05),
    }
    transfer_whole_scheduler_kwargs['total_steps'] = TRANSFER_EPOCHS[1]

    fc_optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=transfer_fc_scheduler_kwargs['max_lr'])
    fc_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(fc_optimizer, **transfer_fc_scheduler_kwargs)

    whole_optimizer = torch.optim.Adam(resnet.parameters(), lr=transfer_whole_scheduler_kwargs['max_lr'])
    whole_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(whole_optimizer, **transfer_whole_scheduler_kwargs)

    test_accuracy = train_CNN_and_get_accuracy(fc_optimizer, fc_lr_scheduler, whole_optimizer, whole_lr_scheduler, resnet, BATCH_SIZE, TRANSFER_EPOCHS)
    return test_accuracy

def train_CNN_and_get_accuracy(fc_optimizer, fc_lr_scheduler, whole_optimizer, whole_lr_scheduler, model, BATCH_SIZE, TRANSFER_EPOCHS):
    transfer_pytorch_model = pytorch_progress.pytorch_model(model, BATCH_SIZE, True)
    transfer_pytorch_model.train(fc_optimizer, fc_lr_scheduler, True, TRANSFER_EPOCHS[0])

    model_fc_trained = transfer_pytorch_model.get_model()
    for param in model_fc_trained.parameters():
        param.requires_grad = True

    transfer_pytorch_model.train(whole_optimizer, whole_lr_scheduler, True, TRANSFER_EPOCHS[1])
    transfer_pytorch_model.test()
    return transfer_pytorch_model.get_test_accuracy()

def CNN_hyperparameter_optimization():
    since = time.time()
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
    study.optimize(CNN_hyperparameters_objective, n_trials = TRIALS, n_jobs=1)
    time_elapsed = time.time() - since
    print(f'Test model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print('='*30)
    print("hyperparameters optimization result:")
    print()
    print(f"{TRIALS} trials optimization time in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print('Best trial hyperparameters:', study.best_trial.params)
    print("test accuracy: ", study.best_trial.value)
    print('='*30)

if __name__ == '__main__':
    # SLP_MLP_hyperparameter_optimization()
    CNN_hyperparameter_optimization()