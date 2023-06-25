import argparse
import time
import torch

import os

import torchattacks

# os.environ["CUDA_VISIBLE_DEVICES"]="7"
from models.models import CNN7, CNNa, CNNb, CNNc, FCModel
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datasets import load_cifar10, load_mnist

# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
import yaml
import mlflow
from test import test_robust
from utils import ParamScheduler, RecursiveNamespace

torch.manual_seed(123123)


def get_param_groups(model, cfg):
    # create 2 parameter groups, one for Bernstein layers and one for others
    if not hasattr(cfg.TRAIN, "BERN_WEIGHT_DECAY"):
        cfg.TRAIN.BERN_WEIGHT_DECAY = cfg.TRAIN.WEIGHT_DECAY
    bernstein_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "bern" in name:
            bernstein_params.append(param)
        else:
            other_params.append(param)
    param_groups = [
        {"params": bernstein_params, "weight_decay": cfg.TRAIN.BERN_WEIGHT_DECAY},
        {"params": other_params, "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
    ]
    return param_groups


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def update_shared_buffers(model, lirpa_model):
    with torch.no_grad():
        for i, (name, buffer) in enumerate(model.named_buffers()):
            lirpa_model.state_dict()[name].copy_(buffer)


def compute_robust_loss(
    model,
    x,
    y,
    eps=None,
    beta=1,
    lirpa_model=None,
    bounding_method="bern",
    num_class=10,
):
    if eps is None:
        return 0.0
    in_lb = torch.maximum(x - eps, torch.zeros_like(x))
    in_ub = torch.minimum(x + eps, torch.ones_like(x))
    c = get_spec(x, y, num_class)
    # yy = model(x[0])
    if bounding_method == "bern":
        inf_ball = torch.cat((in_lb.unsqueeze(-1), in_ub.unsqueeze(-1)), -1)
        out_bounds = model.forward_subinterval(inf_ball, C=c)
        out_lb = out_bounds[..., 0]
        out_ub = out_bounds[..., 1]
    # elif bounding_method == "IBP" and lirpa_model is not None:
    #     ptb = PerturbationLpNorm(norm=np.inf, x_L=in_lb, x_U=in_ub)
    #     my_input = BoundedTensor(x, ptb)
    #     out_lb, out_ub = lirpa_model.compute_bounds(x=(my_input,), C=c, method="IBP")
    # elif bounding_method == "bern-IBP":
    #     inf_ball = torch.cat((in_lb.unsqueeze(-1), in_ub.unsqueeze(-1)), -1)
    #     bern_out_bounds = model.forward_subinterval(inf_ball, C=c)
    #     bern_out_lb = bern_out_bounds[..., 0]
    #     ptb = PerturbationLpNorm(norm=np.inf, x_L=in_lb, x_U=in_ub)
    #     my_input = BoundedTensor(x, ptb)
    #     ibp_out_lb, out_ub = lirpa_model.compute_bounds(
    #         x=(my_input,), C=c, method="IBP"
    #     )
    #     out_lb = beta * bern_out_lb + (1 - beta) * ibp_out_lb
    else:
        raise Exception(
            "Error in Bounding Method. For IBP you have to provide a LIRPA model"
        )
    # lower bound of the target
    if out_lb.shape[-1] == num_class:
        y_one_hot = torch.nn.functional.one_hot(y)
        target_lb = out_lb * y_one_hot
        nontarget_ub = out_ub * (1 - y_one_hot)
        logits = target_lb + nontarget_ub
        labels = y
    else:
        logits = -1 * torch.cat(
            (
                torch.zeros(
                    size=(out_lb.size(0), 1), dtype=out_lb.dtype, device=out_lb.device
                ),
                out_lb,
            ),
            dim=1,
        )
        labels = torch.zeros(
            size=(out_lb.size(0),), dtype=torch.int64, device=out_lb.device
        )

    r_loss = nn.CrossEntropyLoss()(logits, labels)
    return r_loss


def get_spec(x, y, num_class):
    c = torch.eye(num_class).type_as(x)[y].unsqueeze(1) - torch.eye(num_class).type_as(
        x
    ).unsqueeze(0)
    # remove specifications to self
    I = ~(y.data.unsqueeze(1) == torch.arange(num_class).type_as(y.data).unsqueeze(0))
    c = c[I].view(x.size(0), num_class - 1, num_class)
    return c  # minimize ub_other - lb_target


def compute_alpha(min_alpha, epoch, epochs, start_epoch, last_epoch, max_alpha=1.0):
    if epoch < start_epoch:
        return 1.0
    elif epoch >= last_epoch:
        return min_alpha
    else:
        alpha_range = max_alpha - min_alpha
        m = alpha_range / (epochs - start_epoch)
        c = -m * start_epoch
        return max_alpha - (m * epoch + c)


def compute_beta(min_beta, epoch, epochs, max_beta=1.0):
    start_epoch = 0
    if epoch < start_epoch:
        return max_beta
    else:
        beta_range = max_beta - min_beta
        m = beta_range / (epochs - start_epoch)
        c = -m * start_epoch
        return max_beta - (m * epoch + c)


def compute_eps(epoch, epochs, max_eps=1.0):
    max_epochs = int(0.9 * epochs)
    return max_eps * min(1, epoch / max_epochs)


def train(
    model,
    trainloader,
    testloader,
    optimizer,
    loss_fn,
    cfg,
    device="cuda",
    lirpa_model=None,
    start_epoch=0,
    benchmark_loader=None,
):
    best_model_acc = 0
    min_alpha = cfg.ROBUSTNESS.MIN_ALPHA
    max_alpha = cfg.ROBUSTNESS.MAX_ALPHA
    decayRate = cfg.TRAIN.LR_DECAY_RATE
    max_beta = cfg.ROBUSTNESS.MAX_BETA
    min_beta = cfg.ROBUSTNESS.MIN_BETA
    min_eps, max_eps = cfg.ROBUSTNESS.MIN_EPS, cfg.ROBUSTNESS.MAX_EPS
    test_eps = cfg.ROBUSTNESS.TEST_EPS
    epochs = cfg.TRAIN.EPOCHS
    eps_scheduler = ParamScheduler(
        "linear",
        cfg.ROBUSTNESS.EPS_START_EPOCH,
        cfg.ROBUSTNESS.EPS_LAST_EPOCH,
        min_eps,
        max_eps,
    )
    alpha_scheduler = ParamScheduler(
        "linear",
        cfg.ROBUSTNESS.ROBUST_TRAINING_START_EPOCH,
        cfg.ROBUSTNESS.ROBUST_TRAINING_LAST_EPOCH,
        max_alpha,
        min_alpha,
    )
    beta_scheduler = ParamScheduler(
        "linear",
        cfg.ROBUSTNESS.ROBUST_TRAINING_START_EPOCH,
        cfg.ROBUSTNESS.ROBUST_TRAINING_LAST_EPOCH,
        max_beta,
        min_beta,
    )
    lr_decay_start_epoch = (
        30 if cfg.TRAIN.LR_DECAY_START_EPOCH is None else cfg.TRAIN.LR_DECAY_START_EPOCH
    )
    bounding_method = cfg.ROBUSTNESS.BOUNDING_METHOD
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decayRate
    )
    # init forward pass
    with torch.no_grad():
        dummy = model(next(iter(trainloader))[0].to(device))
    num_class = dummy.shape[-1]
    epoch_times = []
    for epoch in range(start_epoch, epochs):
        epoch_s_time = time.perf_counter()
        model.train()
        eps = eps_scheduler.step(epoch)
        with torch.no_grad():
            beta = 0
            alpha = 1.0
            if epoch >= cfg.ROBUSTNESS.WARMUP_EPOCHS and cfg.ROBUSTNESS.ENABLE:
                alpha = alpha_scheduler.step(epoch)
                if "IBP" in bounding_method:
                    beta = beta_scheduler.step(epoch)

        epoch_loss = 0
        epoch_robust_loss = 0

        if cfg.TRAIN.MODE == "adv":
            adversary = torchattacks.PGD(model, eps, alpha=eps / 50, steps=10)
        for batch_idx, (x, y) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            if cfg.TRAIN.MODE == "adv" and epoch >= cfg.ROBUSTNESS.WARMUP_EPOCHS:
                x = adversary(x, y)
            pred = model(x)
            if lirpa_model is not None:
                update_shared_buffers(model, lirpa_model)
            if alpha != 1.0 and cfg.ROBUSTNESS.ENABLE:
                robust_loss = compute_robust_loss(
                    model,
                    x,
                    y,
                    eps,
                    beta=beta,
                    lirpa_model=lirpa_model,
                    bounding_method=bounding_method,
                    num_class=num_class,
                )
            else:
                robust_loss = torch.tensor(0)
            tr_loss = loss_fn(pred, y)
            loss = alpha * tr_loss + (1 - alpha) * robust_loss
            epoch_loss += loss
            epoch_robust_loss += robust_loss
            loss.backward()
            optimizer.step()
            # if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(trainloader):
            #     print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
            #         epoch+1, batch_idx+1, loss))
            with torch.no_grad():
                model(x[0:1])  # Dummy forward to update bounds

        epoch_e_time = time.perf_counter()
        print(
            f"Epoch ({epoch+1}/ {epochs}): Training loss = {epoch_loss}, Robust loss = {epoch_robust_loss}, lr = {optimizer.param_groups[0]['lr']}, Alpha = {alpha:.4f}, Beta = {beta:.4f}, Eps = {eps:.4f}"
        )
        if mlflow_enable:
            mlflow.log_metrics(
                {"lr": optimizer.param_groups[0]["lr"], "Alpha": alpha, "Eps": eps},
                step=epoch + 1,
            )
            mlflow.log_metrics(
                {
                    "Training loss": epoch_loss.item(),
                    "Robust loss": epoch_robust_loss.item(),
                },
                step=epoch + 1,
            )

        total_weights_norm = 0
        with torch.no_grad():
            for p in model.parameters():
                param_norm = p.grad.data.norm(float("inf"))
                total_weights_norm += param_norm.item()
        print(f"Parameters inf-norm = {total_weights_norm}")
        if epoch > lr_decay_start_epoch:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_robust_loss)
            else:
                lr_scheduler.step()

        if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
            # train acc
            correct_cnt = 0
            total_cnt = 0
            total_loss = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(tqdm(trainloader)):
                    x, target = x.to(device), target.to(device)
                    out = model(x)
                    tr_loss = loss_fn(out, target)
                    total_loss += tr_loss
                    _, pred_label = torch.max(out.data, 1)
                    total_cnt += x.shape[0]
                    correct_cnt += (pred_label == target.data).sum()
                model_acc = correct_cnt * 100 / total_cnt
                print(f"Train accuracy: {model_acc}")
                # testing
                if (epoch + 1) % cfg.ROBUSTNESS.TEST_EVERY_N_EPOCH == 0:
                    # torch.cuda.empty_cache()
                    test_acc, cert_acc = test_robust(
                        model,
                        testloader,
                        device=device,
                        eps=test_eps,
                        mode=cfg.TRAIN.MODE,
                    )
                    if benchmark_loader is not None:
                        _, cert_acc = test_robust(
                            model,
                            benchmark_loader,
                            device=device,
                            eps=test_eps,
                            mode=cfg.TRAIN.MODE,
                        )
                    if mlflow_enable:
                        mlflow.log_metrics(
                            {
                                "Test Accuracy": test_acc.item(),
                                "Certified Accuracy": cert_acc,
                            },
                            step=epoch + 1,
                        )
                    if cert_acc >= best_model_acc:
                        best_model_acc = cert_acc
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": tr_loss,
                            },
                            f"{cfg.BASE_DIR}/checkpoint_best_model.pth",
                        )
                        if mlflow_enable:
                            mlflow.log_artifact(
                                f"{cfg.BASE_DIR}/checkpoint_best_model.pth"
                            )
                            mlflow.log_metric(
                                "Max Certified Accuracy", best_model_acc, step=epoch + 1
                            )
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": tr_loss,
                },
                f"{cfg.BASE_DIR}/checkpoint_{epoch+1}.pth",
            )
        epoch_time = epoch_e_time - epoch_s_time
        epoch_times.append(epoch_time)
    epoch_times = np.array(epoch_times)
    print(f"Average epoch time: {epoch_times.mean():.2f} s +-{epoch_times.std():.2f} s")
    if mlflow_enable:
        mlflow.log_metric("Average epoch time", epoch_times.mean(), step=epoch + 1)
        mlflow.log_metric("Std epoch time", epoch_times.std(), step=epoch + 1)


if __name__ == "__main__":
    # num_inputs = 784
    # num_outs = 10
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument("--degree", type=int, help="Degree of the Bernstein polynomial")
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train the model for"
    )
    parser.add_argument(
        "--mlflow",
        type=int,
        help="Enables mlflow experiment tracking",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    # assert len(sys.argv) == 2, "Please provide a YAML configuration file as argument"
    yaml_cfg_f = args.config
    with open(yaml_cfg_f, "rb") as f:
        cfg = RecursiveNamespace(**yaml.safe_load(f))
    # overwrite device in cfg if provided as argument
    if args.device:
        cfg.TRAIN.DEVICE = args.device
    # overwrite degree in cfg if provided as argument
    if args.degree:
        cfg.MODEL.DEGREE = args.degree
    # overwrite epochs in cfg if provided as argument
    if args.epochs:
        cfg.TRAIN.EPOCHS = args.epochs
    degree = cfg.MODEL.DEGREE
    device = cfg.TRAIN.DEVICE
    mlflow_enable = args.mlflow
    # CONFIG = {'checkpoint_path' : "cifar_BN"}
    BASE_DIR = os.path.join(
        cfg.CHECKPOINT.DIR, cfg.EXPERIMENT.NAME, cfg.EXPERIMENT.RUN_NAME
    )
    if mlflow_enable:
        exp = mlflow.get_experiment_by_name(name=cfg.EXPERIMENT.NAME)
        if not exp:
            exp_id = mlflow.create_experiment(cfg.EXPERIMENT.NAME)
        else:
            exp_id = exp.experiment_id

        run = mlflow.start_run(
            run_name=cfg.EXPERIMENT.NAME + "/" + cfg.EXPERIMENT.RUN_NAME,
            experiment_id=exp_id,
        )
        mlflow.log_artifact(yaml_cfg_f)
        mlflow.log_params(cfg.cfg_dict)
    cfg.BASE_DIR = BASE_DIR
    os.makedirs(BASE_DIR, exist_ok=True)
    # batch_size = 512

    is_FC_model = cfg.MODEL.TYPE == "FC"
    if cfg.DATASET == "cifar10":
        trainloader, testloader = load_cifar10(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model
        )
        _, benchmark_testloader = load_cifar10(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model
        )
    elif cfg.DATASET == "mnist":
        trainloader, testloader = load_mnist(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model
        )
        _, benchmark_testloader = load_mnist(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model
        )

    print("==>>> Trainig set size = {}".format(len(trainloader.dataset)))
    print("==>>> Test set size = {}".format(len(testloader.dataset)))
    print(
        "==>>> Robustness Test set size = {}".format(len(benchmark_testloader.dataset))
    )

    in_shape = torch.tensor(next(iter(trainloader))[0][0].shape)
    num_outs = len(trainloader.dataset.classes)
    # layer_sizes = [num_inputs, 512,256,128,64, num_outs]
    if cfg.MODEL.TYPE == "FC":
        layer_sizes = [in_shape.item()] + cfg.MODEL.HIDDEN_LAYERS + [num_outs]
        in_bounds = torch.concat(
            (torch.zeros(in_shape, 1), torch.ones(in_shape, 1)), dim=-1
        ).to(device)
        model = FCModel(
            layer_sizes,
            degree=cfg.MODEL.DEGREE,
            act=cfg.MODEL.ACTIVATION,
            input_bounds=in_bounds,
        ).to(device)
    elif cfg.MODEL.TYPE == "CNNa":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNNa(
            degree, input_bounds=in_bounds, act=cfg.MODEL.ACTIVATION, num_outs=num_outs
        ).to(device)
    elif cfg.MODEL.TYPE == "CNNb":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNNb(
            degree, input_bounds=in_bounds, act=cfg.MODEL.ACTIVATION, num_outs=num_outs
        ).to(device)
    elif cfg.MODEL.TYPE == "CNNc":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNNc(
            degree, input_bounds=in_bounds, act=cfg.MODEL.ACTIVATION, num_outs=num_outs
        ).to(device)
    elif cfg.MODEL.TYPE == "CNN7":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNN7(
            degree, input_bounds=in_bounds, act=cfg.MODEL.ACTIVATION, num_outs=num_outs
        ).to(device)

    print(model)
    init_lr = float(cfg.TRAIN.INIT_LR)
    w_decay = float(cfg.TRAIN.WEIGHT_DECAY)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if mlflow_enable:
        mlflow.log_param("Number of Parameters", num_params)
    print("Parameters:", num_params)
    optim_params = get_param_groups(model, cfg)
    if cfg.TRAIN.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(optim_params, lr=init_lr, weight_decay=w_decay)
    elif cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = optim.Adam(optim_params, lr=init_lr, weight_decay=w_decay)
    else:
        optimizer = optim.SGD(optim_params, lr=init_lr, weight_decay=w_decay)

    start_epoch = 0
    if cfg.CHECKPOINT.LOAD:
        # Load model
        if cfg.CHECKPOINT.PATH_TO_CKPT:
            ckpt = torch.load(cfg.CHECKPOINT.PATH_TO_CKPT)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
        else:
            ckpt = torch.load(f"{BASE_DIR}/checkpoint_best_model.pth")
            # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model_state_dict"])

    bounding_method = cfg.ROBUSTNESS.BOUNDING_METHOD
    lirpa_model = None
    # if "IBP" in bounding_method:
    #     model.eval()
    #     dummy_input = next(iter(trainloader))[0]
    #     lirpa_model = BoundedModule(model, dummy_input)
    #     model.train()

    criterion = nn.CrossEntropyLoss()
    train(
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        cfg,
        device=device,
        lirpa_model=lirpa_model,
        start_epoch=start_epoch,
        benchmark_loader=benchmark_testloader,
    )
    data_point = trainloader.dataset[1][0].to(device)
    model(data_point.unsqueeze(0))
