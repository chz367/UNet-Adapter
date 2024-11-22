import torch
import argparse
import os
import shutil
import h5py
import numpy as np
from tqdm import tqdm
from network.unet_urpc_adapter import UNet_URPC_Adapter
import shutil
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='../model/ACDC_RLV/Uncertainty_Rectified_Pyramid_Consistency_140_labeled/unet_urpc_old_adapter/unet_urpc_old_adapter_best_model.pth',
                    help='Path to the trained model file')
parser.add_argument('--root_path', type=str, default='../data/ACDC-RLV', help='Root path of the dataset')
parser.add_argument('--exp', type=str, default='ACDC/Uncertainty_Rectified_Pyramid_Consistency', help='Experiment name')
parser.add_argument('--model', type=str, default='unet_urpc_adapter', help='Model name')

parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--labeled_num', type=int, default=140, help='Number of labeled data')  # 原值7
args = parser.parse_args()

def load_model(model_path, num_classes):
    net = UNet_URPC_Adapter(in_chns=1, class_num=num_classes)

    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    return net

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # 使用unsqueeze添加两个维度
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            # print("chz type test:",type(out_main))
            out_main = out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)   # 右心室
    second_metric = calculate_metric_percase(prediction == 2, label == 2)  # 左心室

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return first_metric, second_metric


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    # if np.sum(pred) == 0 or np.sum(gt) == 0:
    #     return 1,0,0

    asd = metric.binary.asd(pred, gt)
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)

    return dice, hd95, asd

def crop_or_resize(image, target_shape):
    resized_image = zoom(image, (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]), order=0)
    return resized_image


def test_model(model, root_path, exp, labeled_num, num_classes):
    test_save_path = "..\\test\{}_{}_labeled\{}_predictions\\".format(exp, labeled_num, args.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    print('chz test:', root_path)
    # with open(os.path.join(root_path, 'val.list'), 'r') as f:
    with open(os.path.join(root_path, 'test.list'), 'r') as f:

        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    print(image_list)
    print("---------------------image_list-----------------------------")
    first1 = 0
    first2 = 0
    first3 = 0
    second1 = 0
    second2 = 0
    second3 = 0
    # first_metric1=(0,0,0)
    # second_metric1=(0,0,0)
    dice_sum_list = []
    hd_sum_list = []
    asd_sum_list = []
    number_ = len(image_list)
    for case in tqdm(image_list):
        print(type(case))
        print("---------------------case-----------------------------")
        print(case)
        first_metric, second_metric = test_single_volume(case, model, test_save_path, args)
        # first_metric1 = first_metric1 + first_metric
        # second_metric1 = second_metric1 + second_metric
        first1 = first1 + first_metric[0]
        first2 = first2 + first_metric[1]
        first3 = first3 + first_metric[2]

        second1 = second1 + second_metric[0]
        second2 = second2 + second_metric[1]
        second3 = second3 + second_metric[2]

        print("first_metric:", first_metric)
        print("second_metric:", second_metric)
        dice_sum = (first_metric[0] + second_metric[0])/2
        hd_sum = (first_metric[1] + second_metric[1])/2
        asd_sum = (first_metric[2] + second_metric[2])/2

        dice_sum_list.append(dice_sum)
        hd_sum_list.append(hd_sum)
        asd_sum_list.append(asd_sum)
    print("first_metric2:", (first1/number_, first2/number_, first3/number_))
    print("second_metric2:", (second1/number_, second2/number_, second3/number_))
    print("metrics", (first1/number_+second1/number_)/2, (first2/number_+second2/number_)/2, (first3/number_+second3/number_)/2)

    print("dice_data =", dice_sum_list)
    print("hd95_data =", hd_sum_list)
    print("asd_data =", asd_sum_list)


if __name__ == "__main__":
    model_path = args.model_path
    if 'iter_' in model_path:
        iter_num = int(model_path.split('/')[-1].split('_')[1])
        model = load_model(model_path, args.num_classes)
    elif 'best_model' in model_path:
        model = load_model(model_path, args.num_classes)
    else:
        raise ValueError("Invalid model path. Please provide a valid model file.")

    test_model(model, args.root_path, args.exp, args.labeled_num, args.num_classes)

