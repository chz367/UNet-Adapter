import os
import pickle
import numpy as np
import re
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

def load_model(path):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        for key in checkpoint["state_dict"]:
            print(key)

        N = checkpoint["state_dict"]["decoder.out_conv.bias"].size()

        sob = "sobel.0.weight" in checkpoint["state_dict"].keys()
        model = models.__dict__[checkpoint["arch"]](sobel=sob, out=int(N[0]))

        def rename_key(key):
            if not "module" in key:
                return key
            return "".join(key.split(".module"))

        checkpoint["state_dict"] = {
            rename_key(key): val for key, val in checkpoint["state_dict"].items()
        }

        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


def load_checkpoint(path, model, optimizer, from_ddp=False):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    return model, optimizer, checkpoint["epoch"], loss.item()


def restore_model(logger, snapshot_path, model_num=None):
    try:
        logger.info(f"Snapshot path: {snapshot_path}")
        iter_num = []
        name = "model_iter"
        if model_num:
            name = model_num
        for filename in os.listdir(snapshot_path):
            if name in filename:
                basename, extension = os.path.splitext(filename)
                iter_num.append(int(basename.split("_")[2]))
        iter_num = max(iter_num)
        for filename in os.listdir(snapshot_path):
            if name in filename and str(iter_num) in filename:
                model_checkpoint = filename
    except Exception as e:
        logger.warning(f"Error finding previous checkpoints: {e}")

    try:
        logger.info(f"Restoring model checkpoint: {model_checkpoint}")
        model, optimizer, start_epoch, performance = load_checkpoint(
            snapshot_path + "/" + model_checkpoint, model, optimizer
        )
        logger.info(f"Models restored from iteration {iter_num}")
        return model, optimizer, start_epoch, performance
    except Exception as e:
        logger.warning(f"Unable to restore model checkpoint: {e}, using new model")


def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


class UnifLabelSampler(Sampler):
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel),
            )
            res[i * size_per_pseudolabel : (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[: self.N].astype("int")

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group["weight_decay"] * t)
        param_group["lr"] = lr


class Logger:
    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), "wb") as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode="inner").astype(
                np.uint8
            )
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (
                posdis - np.min(posdis)
            ) / (np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
    return normalized_sdf


def distributed_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print("setting up dist process group now")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_ddp_to_nddp(state_dict):
    pattern = re.compile("module")
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict = state_dict
    return model_dict
