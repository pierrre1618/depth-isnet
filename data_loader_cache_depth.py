## data loader
## Ackownledgement:
## We would like to thank Dr. Ibrahim Almakky (https://scholar.google.co.uk/citations?user=T9MTcK0AAAAJ&hl=en)
## for his helps in implementing cache machanism of our DIS dataloader.
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
import json
from tqdm import tqdm
from skimage import io
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import random
from transformers import pipeline
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor



#### --------------------- DIS dataloader cache ---------------------####


def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []
    
    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], "<<<---")
        
        # Get the list of image file paths.
        tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + '*' + datasets[i]["im_ext"])
        print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))
        
        # Build the ground truth file list.
        if datasets[i]["gt_dir"] == "":
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                os.path.join(
                    datasets[i]["gt_dir"],
                    x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i]["gt_ext"]
                )
                for x in tmp_im_list
            ]
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))
        
        # Build the depth file list.
        if "depth_dir" not in datasets[i] or datasets[i]["depth_dir"] == "":
            print('-depth-', datasets[i]["name"], ': ', 'No Depth Found')
            tmp_depth_list = []
        else:
            tmp_depth_list = [
                os.path.join(
                    datasets[i]["depth_dir"],
                    x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i]["depth_ext"]
                )
                for x in tmp_im_list
            ]
            print('-depth-', datasets[i]["name"], datasets[i]["depth_dir"], ': ', len(tmp_depth_list))
        
        # Combine multiple training sets into one dataset if flag=="train"
        if flag == "train":
            if len(name_im_gt_list) == 0:
                name_im_gt_list.append({
                    "dataset_name": datasets[i]["name"],
                    "im_path": tmp_im_list,
                    "gt_path": tmp_gt_list,
                    "depth_path": tmp_depth_list,
                    "im_ext": datasets[i]["im_ext"],
                    "gt_ext": datasets[i]["gt_ext"],
                    "depth_ext": datasets[i].get("depth_ext", ""),
                    "cache_dir": datasets[i]["cache_dir"]
                })
            else:
                name_im_gt_list[0]["dataset_name"] = name_im_gt_list[0]["dataset_name"] + "_" + datasets[i]["name"]
                name_im_gt_list[0]["im_path"] += tmp_im_list
                name_im_gt_list[0]["gt_path"] += tmp_gt_list
                name_im_gt_list[0]["depth_path"] += tmp_depth_list

                # Check format and update the cache_dir if needed.
                if datasets[i]["im_ext"] != ".jpg" or datasets[i]["gt_ext"] != ".png":
                    print("Error: Please make sure all your images and ground truth masks are in jpg and png format respectively !!!")
                    exit()
                name_im_gt_list[0]["im_ext"] = ".jpg"
                name_im_gt_list[0]["gt_ext"] = ".png"
                name_im_gt_list[0]["cache_dir"] = os.sep.join(datasets[i]["cache_dir"].split(os.sep)[0:-1]) \
                                                  + os.sep + name_im_gt_list[0]["dataset_name"]
        else:
            # For validation or inference, keep each dataset separate.
            name_im_gt_list.append({
                "dataset_name": datasets[i]["name"],
                "im_path": tmp_im_list,
                "gt_path": tmp_gt_list,
                "depth_path": tmp_depth_list,
                "im_ext": datasets[i]["im_ext"],
                "gt_ext": datasets[i]["gt_ext"],
                "depth_ext": datasets[i].get("depth_ext", ""),
                "cache_dir": datasets[i]["cache_dir"]
            })
    
    return name_im_gt_list

def create_dataloaders(name_im_gt_list, cache_size=[], cache_boost=True, my_transforms=[], batch_size=1, shuffle=False,pipe=None):
    ## model="train": return one dataloader for training
    ## model="valid": return a list of dataloaders for validation or testing

    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    for i in range(0,len(name_im_gt_list)):
        gos_dataset = GOSDatasetCache([name_im_gt_list[i]],
                                      cache_size = cache_size,
                                      cache_path = name_im_gt_list[i]["cache_dir"],
                                      cache_boost = cache_boost,
                                      transform = transforms.Compose(my_transforms),
                                      pipe=pipe
                                      )
        gos_dataloaders.append(DataLoader(gos_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers_))
        gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

def im_reader(im_path):
    return io.imread(im_path)

def im_preprocess(im,size):
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
    if(len(size)<2):
        return im_tensor, im.shape[0:2]
    else:
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = F.upsample(im_tensor, size, mode="bilinear")
        im_tensor = torch.squeeze(im_tensor,0)

    return im_tensor.type(torch.uint8), im.shape[0:2]

def gt_preprocess(gt,size):
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8),0)

    if(len(size)<2):
        return gt_tensor.type(torch.uint8), gt.shape[0:2]
    else:
        gt_tensor = torch.unsqueeze(torch.tensor(gt_tensor, dtype=torch.float32),0)
        gt_tensor = F.upsample(gt_tensor, size, mode="bilinear")
        gt_tensor = torch.squeeze(gt_tensor,0)

    return gt_tensor.type(torch.uint8), gt.shape[0:2]
    # return gt_tensor, gt.shape[0:2]


class GOSColorEnhanceTransform(object):
    def __init__(self, brightness_range=(0.7, 1.2), prob=0.5, 
                 contrast_range=(0.7, 1.2), color_range=(0.0, 2.0), sharpness_range=(0.0, 3.0)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.color_range = color_range
        self.sharpness_range = sharpness_range
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # Apply color enhancements with probability (if random >= prob then enhance)
        if random.random() >= self.prob:
            image = self.color_enhance(image)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}

    def color_enhance(self, image):
        # Brightness adjustment
        bright_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = self.adjust_brightness(image, bright_factor)
        
        # Contrast adjustment
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        image = self.adjust_contrast(image, contrast_factor)
        
        # Color adjustment
        color_factor = random.uniform(self.color_range[0], self.color_range[1])
        image = self.adjust_color(image, color_factor)

        # Sharpness adjustment is more complex in PyTorch, so it is skipped here.
        return image

    def adjust_brightness(self, image, factor):
        # If image has 4 channels, assume the first 3 are RGB and apply brightness to them.
        if image.shape[0] == 4:
            rgb = torch.clamp(image[:3] * factor, 0.0, 1.0)
            extra = image[3:].clone()  # Keep the extra channel unchanged
            return torch.cat([rgb, extra], dim=0)
        else:
            return torch.clamp(image * factor, 0.0, 1.0)

    def adjust_contrast(self, image, factor):
        if image.shape[0] == 4:
            rgb = image[:3]
            extra = image[3:].clone()
            mean = torch.mean(rgb, dim=(1, 2), keepdim=True)
            rgb = torch.clamp((rgb - mean) * factor + mean, 0.0, 1.0)
            return torch.cat([rgb, extra], dim=0)
        else:
            mean = torch.mean(image, dim=(1, 2), keepdim=True)
            return torch.clamp((image - mean) * factor + mean, 0.0, 1.0)

    def adjust_color(self, image, factor):
        if image.shape[0] == 4:
            rgb = image[:3]
            extra = image[3:].clone()
            grayscale = torch.mean(rgb, dim=0, keepdim=True)
            rgb_adjusted = torch.clamp(grayscale + (rgb - grayscale) * factor, 0.0, 1.0)
            return torch.cat([rgb_adjusted, extra], dim=0)
        else:
            grayscale = torch.mean(image, dim=0, keepdim=True)
            return torch.clamp(grayscale + (image - grayscale) * factor, 0.0, 1.0)




class GOSRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class GOSResize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # import time
        # start = time.time()

        image = torch.squeeze(F.upsample(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.upsample(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        # print("time for resize: ", time.time()-start)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class GOSRandomCrop(object):
    def __init__(self,size=[288,288],prob=0.5):
        self.size = size
        self.prob = prob

    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        if random.random() >= self.prob:
            h, w = image.shape[1:]
            new_h, new_w = self.size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[:,top:top+new_h,left:left+new_w]
            label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}


class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406,0.406], std=[0.229,0.224,0.225,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        #print(" image=",image.shape," mean=",self.mean," std=",self.std)
        image = normalize(image,self.mean,self.std)


        #plt.figure()
        #plt.imshow(np.clip(image[:3].permute(1, 2, 0).numpy(), 0, 1))
        #plt.show()

        #plt.figure()
        #plt.imshow(np.clip(image[3].numpy(), 0, 1))
        #plt.show()

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}


class GOSDatasetCache(Dataset):

    def __init__(self, name_im_gt_list, cache_size=[], cache_path='./cache', cache_file_name='dataset.json', cache_boost=False, transform=None,pipe=None):


        self.cache_size = cache_size
        self.cache_path = cache_path
        self.cache_file_name = cache_file_name
        self.cache_boost_name = ""
        self.pipe=pipe
        self.cache_boost = cache_boost
        # self.ims_npy = None
        # self.gts_npy = None

        ## cache all the images and ground truth into a single pytorch tensor
        self.ims_pt = None
        self.gts_pt = None

        ## we will cache the npy as well regardless of the cache_boost
        # if(self.cache_boost):
        self.cache_boost_name = cache_file_name.split('.json')[0]

        self.transform = transform

        self.dataset = {}

        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_shp"] = []
        self.dataset["gt_shp"] = []
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list


        self.dataset["ims_pt_dir"] = ""
        self.dataset["gts_pt_dir"] = ""

        self.dataset = self.manage_cache(dataset_names)

    def manage_cache(self,dataset_names):
        if not os.path.exists(self.cache_path): # create the folder for cache
            os.makedirs(self.cache_path)
        cache_folder = os.path.join(self.cache_path, "_".join(dataset_names)+"_"+"x".join([str(x) for x in self.cache_size]))
        if not os.path.exists(cache_folder): # check if the cache files are there, if not then cache
            return self.cache(cache_folder)
        return self.load_cache(cache_folder)

    def cache(self, cache_folder):
        # Create the cache folder if it doesn't exist.
        os.makedirs(cache_folder, exist_ok=True)
        cached_dataset = deepcopy(self.dataset)

        # Lists to hold aggregated tensors (if cache_boost is enabled)
        ims_pt_list = []
        gts_pt_list = []
        depth_pt_list = []  # for depth images

        # Make sure the keys for shapes and paths exist.
        if "im_shp" not in cached_dataset:
            cached_dataset["im_shp"] = []
        if "gt_shp" not in cached_dataset:
            cached_dataset["gt_shp"] = []
        # Prepare new keys for depth information.
        cached_dataset["depth_path"] = []
        cached_dataset["depth_shp"] = []
        # Create the depth estimation pipeline once.

        for i, im_path in tqdm(enumerate(self.dataset["im_path"]),
                                 total=len(self.dataset["im_path"])):
            im_id = cached_dataset["im_name"][i]
            print("Processing im_path:", im_path)
            # Read and pre-process the image.
            im = im_reader(im_path)
            im, im_shp = im_preprocess(im, self.cache_size)

            # Save the processed image.
            im_cache_file = os.path.join(
                cache_folder, f"{self.dataset['data_name'][i]}_{im_id}_im.pt")
            torch.save(im, im_cache_file)
            cached_dataset["im_path"][i] = im_cache_file
            if self.cache_boost:
                ims_pt_list.append(torch.unsqueeze(im, 0))

            # Process the ground-truth.
            gt = np.zeros(im.shape[1:3])  # default ground truth if none provided
            if len(self.dataset["gt_path"]) != 0:
                gt = im_reader(self.dataset["gt_path"][i])
            gt, gt_shp = gt_preprocess(gt, self.cache_size)
            gt_cache_file = os.path.join(
                cache_folder, f"{self.dataset['data_name'][i]}_{im_id}_gt.pt")
            torch.save(gt, gt_cache_file)
            # Update the gt path in the cached dataset.
            if len(self.dataset["gt_path"]) > 0:
                cached_dataset["gt_path"][i] = gt_cache_file
            else:
                cached_dataset["gt_path"].append(gt_cache_file)
            if self.cache_boost:
                gts_pt_list.append(torch.unsqueeze(gt, 0))

            # Save image and gt shape information.
            cached_dataset["im_shp"].append(im_shp)
            cached_dataset["gt_shp"].append(gt_shp)

            # ======= Process the Depth Image =======
            # Create the depth image using create_depth.
            depth = self.create_depth(im)
            # (Optional) If you need to preprocess depth or extract its shape:
            # depth, depth_shp = depth_preprocess(depth, self.cache_size)
            depth_shp = depth.shape  # Here, we simply store the tensor shape.
            depth_cache_file = os.path.join(
                cache_folder, f"{self.dataset['data_name'][i]}_{im_id}_depth.pt")
            torch.save(depth, depth_cache_file)
            # Update the cached dataset with depth info.
            cached_dataset["depth_path"].append(depth_cache_file)
            cached_dataset["depth_shp"].append(depth_shp)
            if self.cache_boost:
                depth_pt_list.append(torch.unsqueeze(depth, 0))
            # =======================================

        # If caching in boost mode, aggregate the individual tensors.
        if self.cache_boost:
            cached_dataset["ims_pt_dir"] = os.path.join(
                cache_folder, self.cache_boost_name + '_ims.pt')
            cached_dataset["gts_pt_dir"] = os.path.join(
                cache_folder, self.cache_boost_name + '_gts.pt')
            cached_dataset["depth_pt_dir"] = os.path.join(
                cache_folder, self.cache_boost_name + '_depth.pt')
            self.ims_pt = torch.cat(ims_pt_list, dim=0)
            self.gts_pt = torch.cat(gts_pt_list, dim=0)
            self.depth_pt = torch.cat(depth_pt_list, dim=0)
            torch.save(self.ims_pt, cached_dataset["ims_pt_dir"])
            torch.save(self.gts_pt, cached_dataset["gts_pt_dir"])
            torch.save(self.depth_pt, cached_dataset["depth_pt_dir"])

        # Save the updated dataset metadata as a JSON file.
        try:
            with open(os.path.join(cache_folder, self.cache_file_name), "w") as json_file:
                json.dump(cached_dataset, json_file)
        except Exception as e:
            raise FileNotFoundError("Cannot create JSON") from e

        return cached_dataset

    def load_cache(self, cache_folder):
        with open(os.path.join(cache_folder, self.cache_file_name), "r") as json_file:
            dataset = json.load(json_file)
        # If using cache_boost, load the aggregated tensors into memory.
        if self.cache_boost:
            self.ims_pt = torch.load(dataset["ims_pt_dir"], map_location='cpu')
            self.gts_pt = torch.load(dataset["gts_pt_dir"], map_location='cpu')
            self.depth_pt = torch.load(dataset["depth_pt_dir"], map_location='cpu')
        return dataset

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im = None
        gt = None
        depth = None

        if self.cache_boost and self.ims_pt is not None:
            # Load image, ground truth, and depth from cached tensors in memory.
            im = self.ims_pt[idx]
            gt = self.gts_pt[idx]
            depth = self.depth_pt[idx]
        else:
            # Load image from cache file on disk.
            im_pt_path = os.path.join(self.cache_path, os.sep.join(self.dataset["im_path"][idx].split(os.sep)[-2:]))
            im = torch.load(im_pt_path)
            
            # Load ground truth from cache file on disk.
            gt_pt_path = os.path.join(self.cache_path, os.sep.join(self.dataset["gt_path"][idx].split(os.sep)[-2:]))
            gt = torch.load(gt_pt_path)
            
            # Load depth image from cache file on disk.
            depth_pt_path = os.path.join(self.cache_path, os.sep.join(self.dataset["depth_path"][idx].split(os.sep)[-2:]))
            depth = torch.load(depth_pt_path)
        
        # Get the original image shape (if needed for later processing)
        im_shp = self.dataset["im_shp"][idx]
        
        # Normalize image, ground truth, and depth to [0, 1]
        im = torch.divide(im, 255.0)
        gt = torch.divide(gt, 255.0)
        depth = torch.divide(depth, 255.0)
        
        # Fuse the depth channel with the image.
        # Assuming 'im' has shape [C, H, W] (e.g., [3, H, W]) and depth has shape [1, H, W]
        fused_im = torch.cat([im, depth], dim=0)
        
        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": fused_im,
            "label": gt,
            "shape": torch.from_numpy(np.array(im_shp)),
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


    def create_depth(self,im: torch.Tensor) -> torch.Tensor:
        """
        Given an image as a torch tensor (shape [C, H, W] with pixel values [0, 255]),
        this function uses a Hugging Face depth-estimation pipeline to estimate the depth.
        It returns a torch tensor of shape [1, H, W] (with values scaled to [0, 1])
        that is guaranteed to match the input image's spatial dimensions.
        """
        # Convert the input torch tensor to a PIL image.
        to_pil = ToPILImage()
        pil_image = to_pil(im)
        
        # Run the depth estimation pipeline.
        result = self.pipe(pil_image)
        # The pipeline may return a list of dictionaries. Use the first one if needed.
        if isinstance(result, list):
            result = result[0]
        
        # Extract the depth map from the result.
        # The pipeline is expected to return a dict with a "depth" key.
        depth_map = result["depth"]
        
        # Ensure depth_map is a NumPy array.
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map,dtype=np.float32)
        depth_pil = Image.fromarray(np.clip(depth_map,0,255))

        # Get the original image size (width, height) from the PIL image.
        W, H = pil_image.size
        # Resize the depth image if necessary to match the original image dimensions.
        if depth_pil.size != (W, H):
            depth_pil = depth_pil.resize((W, H), resample=Image.BILINEAR)
        
        # Convert the depth PIL image back to a torch tensor.
        depth_tensor=torch.tensor(np.array(depth_pil), dtype=torch.uint8).unsqueeze(0)
        
        return depth_tensor


