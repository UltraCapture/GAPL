import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
from benchmarks import Benchmarks 
from collections import defaultdict
import datasets
import json
import yaml
import models

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def find_best_threshold(y_true, y_pred):
    N = y_true.shape[0]
    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)

def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    

def validate(model, loader, find_thres=False):
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in loader:
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 

def validate_separate(model, loader):
    generator_data = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    
    model.eval()
    
    with torch.no_grad():
        for img, label, generator in loader:
            in_tens = img.cuda() 
            predictions = model(in_tens).sigmoid().flatten().cpu().tolist()
            
            true_labels = label.flatten().tolist()
            
            for pred, lab, gen_key in zip(predictions, true_labels, generator):
                generator_data[gen_key]['y_pred'].append(pred)
                generator_data[gen_key]['y_true'].append(lab)
    results = {}
    
    print("\n--- Generator Validataion ---")
    
    for gen_key, data in generator_data.items():
        y_true = np.array(data['y_true'])
        y_pred = np.array(data['y_pred'])
        
        if len(y_true) == 0 or (y_true.sum() == 0 and (1 - y_true).sum() == 0):
             ap = float('nan')
        else:
             ap = average_precision_score(y_true, y_pred)
    
        r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
        results[gen_key] = {
            'AP': ap,
            'Acc': acc0,
            'R_Acc': r_acc0,
            'F_Acc': f_acc0,
            'Count': len(y_true)
        }
        
        print(f"Generator: {gen_key}")
        print(f"  > AP: {ap:.4f}")
        print(f"  > Acc@0.5: {acc0:.4f}")
        print("-" * 20)
        
    return results


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        opt):

        assert data_mode in ["wang2020", "ours", "GenImage", "Xdet"]
        self.jpeg_quality = opt.jpeg_quality
        self.gaussian_sigma = opt.gaussian_sigma

        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l
        print(f"real {len(real_list)} fake {len(fake_list)}")
        self.total_list = real_list + fake_list

        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(opt.input_size),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])

    def read_path(self, real_path, fake_path, data_mode, max_sample):
        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        # assert len(real_list) == len(fake_list)  
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx): 
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label

def iter_dataset(dataset):
    mode = dataset['mode']
    classes = dataset['classes']
    path = dataset['path']

    if mode == 'Xdet':
        for cls in classes:
            yield dict(
                key = cls,
                real_path = os.path.join(path, cls, 'sample'),
                fake_path = os.path.join(path, cls, 'synthetic'),
                data_mode = 'Xdet'
            )

    elif mode == 'wang':
        for cls in classes:
            yield dict(
                key = cls,
                real_path = os.path.join(path, cls),
                fake_path = os.path.join(path, cls),
                data_mode = 'wang2020'
            )

    elif mode == 'GenImage':
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            for cls in classes:
                if cls in folder:
                    yield dict(
                        key = cls,
                        real_path = os.path.join(folder_path, 'val', 'nature'),
                        fake_path = os.path.join(folder_path, 'val', 'ai'),
                        data_mode = 'GenImage'
                    )
    
    elif mode == "Synthbuster":
        for cls in classes:
            yield dict(
                key = cls
            )
    
    elif mode == "huggingface":
        for cls in classes:
            yield dict(
                key = cls
            )

class SynthbusterDataset(Dataset):
    def __init__(self, path, gen_name, stat="imagenet", opt=None):
        with open(os.path.join(path, "test.json"), "r") as f:
            test = json.load(f)
        with open(os.path.join(path, "mapping.json"), "r") as f:
            mapping = json.load(f)
        
        self.data_list = list()
        self.data = datasets.load_from_disk(path)

        for generator_name, files in test.items():
            if generator_name == gen_name:
                for file_path, label in files.items():
                    file_id = mapping.get(file_path)
                    if file_id is None:
                        print(f"Warning: {file_path} not found in mapping.json")
                        continue
                    self.data_list.append((file_id, label))

        stat_from = "imagenet" if stat.lower().startswith("imagenet") else "clip"

        self.transform = transforms.Compose([
            transforms.CenterCrop(opt.input_size),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        id, label = self.data_list[index]
        image = Image.open(BytesIO(self.data[id]["image"])).convert("RGB")
        image = self.transform(image)
        return image, label

class HfEvalDataset(Dataset):
    def __init__(self, name="OwensLab/CommunityForensics-Eval", stat="imagenet", opt=None):
        super().__init__()
        self.data = datasets.load_dataset(name, split="CompEval", cache_dir="~/.cache/huggingface/datasets")
        stat_from = "imagenet" if stat.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(opt.input_size),
            # transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.data[index]['label']
        image = Image.open(BytesIO(self.data[index]["image_data"])).convert("RGB")
        image = self.transform(image)
        generator_name = self.data[index]['model_name']
        return image, label, generator_name
        # return image, label, 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_folder', type=str, default='results', help='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--model"      , '-m', type=str, help="Model to test", default='BFREE_dino2reg4')
    parser.add_argument("--model_size"      , '-ms', type=str, help="Model size", default='BFREE_dino2reg4')
    parser.add_argument("--input_size"      , '-is', type=int, help="image input size", default='BFREE_dino2reg4')
    parser.add_argument("--patch_size"      , '-ps', type=int, help="image patch size", default='BFREE_dino2reg4')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')
    parser.add_argument('--arch', type=str, default='imagenet')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")
    parser.add_argument("--experiment_name", type=str, default="TEST")
    parser.add_argument('--result_file', type=str, default='result.txt')

    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_folder):
        # shutil.rmtree(opt.result_folder)
        os.makedirs(opt.result_folder)


    model = models.GAPLModel(
        fe_path=None,
        proto_path=None,
        freeze_backbone=False
    )

    state_dict = torch.load(opt.ckpt, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict['model'], strict=False)
    model.load_prototype(state_dict['prototype'])
    print(missing_keys, unexpected_keys)
    print ("Model loaded..")
    
    model.eval()
    model.cuda()

    for dataset in Benchmarks:
        with open(os.path.join(opt.result_folder, opt.result_file), 'a') as f:
            f.write("="*40)
            f.write(f"{dataset['name']}")
            f.write("="*40 + '\n')
    
        accs = []; aps = []

        dataset_mode = dataset['mode']
        path = dataset['path']
        set_seed()
        
        for dataset_path in iter_dataset(dataset):
            if dataset_mode == "Synthbuster":
                dataset = SynthbusterDataset(path, dataset_path['key'])
            elif dataset_mode == "huggingface":
                dataset = HfEvalDataset()
            else:
                dataset = RealFakeDataset(dataset_path['real_path'], 
                                        dataset_path['fake_path'], 
                                        dataset_path['data_mode'], 
                                        opt.max_sample,
                                        # None,
                                        opt.arch,
                                        opt,
                                        )
            
            print(f"Validating {dataset_path['key']} now, length{len(dataset)}".center(60, '='))
            loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
            if dataset_mode == 'huggingface':
                results = validate_separate(model, loader)
                for gen_name, data in results.items():
                    ap, r_acc0, f_acc0, acc0 = data['AP'], data['R_Acc'], data['F_Acc'], data['Acc']
                    print(f"Testing Set:{gen_name:<10} AP: {ap*100:6.2f}  R_Acc: {r_acc0*100:6.2f}  F_Acc: {f_acc0*100:6.2f}  Acc: {acc0*100:6.2f}")
                    with open(os.path.join(opt.result_folder, opt.result_file), 'a') as f:
                        f.write(f"Testing Set:{gen_name:<10} AP: {ap*100:6.2f}  R_Acc: {r_acc0*100:6.2f}  F_Acc: {f_acc0*100:6.2f}  Acc: {acc0*100:6.2f}\n")
                    accs.append(acc0*100)
                    aps.append(ap*100)
            else:
                ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)
                accs.append(acc0*100)
                aps.append(ap*100)
                print(f"Testing Set:{dataset_path['key']:<10} AP: {ap*100:6.2f}  R_Acc: {r_acc0*100:6.2f}  F_Acc: {f_acc0*100:6.2f}  Acc: {acc0*100:6.2f}")
                with open(os.path.join(opt.result_folder, opt.result_file), 'a') as f:
                    f.write(f"Testing Set:{dataset_path['key']:<10} AP: {ap*100:6.2f}  R_Acc: {r_acc0*100:6.2f}  F_Acc: {f_acc0*100:6.2f}  Acc: {acc0*100:6.2f}\n")

        accs = np.mean(np.array(accs))
        aps = np.mean(np.array(aps))
        
        with open(os.path.join(opt.result_folder,opt.result_file), 'a') as f:
            f.write(f"Average acc {round(accs, 2)}")
            f.write(f"Average ap {round(aps, 2)}\n")
            print(f"Average acc {round(accs, 2)}  Average ap {round(aps, 2)}")
    