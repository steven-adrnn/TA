# abstract_dataset
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

import lmdb

sys.path.append('.')

import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as TA_transforms
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from .albu import IsotropicResize

SUKU_DICT = {
    'BAL': 0, 'BJR': 1, 'BTK': 2, 'BUG': 3, 'CIN': 4,
    'DYK': 5, 'JAW': 6, 'MKS': 7, 'MLK': 8, 'MIN': 9,
    'PAP': 10, 'SSK': 11, 'SUN': 12,
    'NON_INDO': 13
}

SUKU_LIST = list(SUKU_DICT.keys())

GLOBAL_LMDB_ENVS = {}
FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT', 'FF-FH', 'IndoDeepfake', 'Indo-MM', 'Indo-FS', 'Indo-VC']
def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        global GLOBAL_LMDB_ENVS
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compressions']
        self.frame_num = config['frame_num'][mode]
        
        self._init_indo_demographic_mapping()

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list)>1:
                    if all_in_pool(dataset_list,FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        # self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                        if lmdb_path not in GLOBAL_LMDB_ENVS:
                            GLOBAL_LMDB_ENVS[lmdb_path] = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                        self.env = GLOBAL_LMDB_ENVS[lmdb_path]
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    if lmdb_path not in GLOBAL_LMDB_ENVS:
                        GLOBAL_LMDB_ENVS[lmdb_path] = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    self.env = GLOBAL_LMDB_ENVS[lmdb_path]
        elif mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'], f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                # self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                if lmdb_path not in GLOBAL_LMDB_ENVS:
                    GLOBAL_LMDB_ENVS[lmdb_path] = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                self.env = GLOBAL_LMDB_ENVS[lmdb_path]
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        
    def _init_indo_demographic_mapping(self):
        """
        Memetakan actor_id (str) -> label suku (int, 0-12).
 
        Membaca dari SEMUA tiga JSON (Indo-MM, Indo-FS, Indo-VC). 
        Hanya membaca real video karena hanya format itu yang 
        mengandung nama suku secara eksplisit:
            001_BAL_M_144_REAL  ->  '001': 0   (BAL = 0)
        """
        self.indo_demographics = {}
 
        json_folder = self.config['dataset_json_folder']
        if not os.path.exists(json_folder):
            json_folder = json_folder.replace(
                '/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public'
            )
 
        for ds_name in ['Indo-MM', 'Indo-FS', 'Indo-VC']:
            json_path = os.path.join(json_folder, f'{ds_name}.json')
            if not os.path.exists(json_path):
                continue
 
            with open(json_path, 'r') as f:
                data = json.load(f)
 
            try:
                real_data = data[ds_name]['Indo-real']
                for mode_key in real_data:            # train, val, test
                    for comp in real_data[mode_key]:  # 720p, 144p
                        for vid_name in real_data[mode_key][comp]:
                            # Format: 001_BAL_M_144_REAL
                            parts = vid_name.split('_')
                            if len(parts) >= 2 and parts[1] in SUKU_DICT:
                                actor_id = parts[0]
                                self.indo_demographics[actor_id] = SUKU_DICT[parts[1]]
            except (KeyError, TypeError) as e:
                print(f"[WARNING] _init_indo_demographic_mapping ({ds_name}): {e}")
 
        print(f"[INFO] Indo demographic mapping: {len(self.indo_demographics)} actor IDs dimuat.")
 
    # ==============================================================
    # Helper untuk ekstrak demographic label
    # ==============================================================
    def _get_demographic_label(self, video_name, file_path):
        """
        Mengekstrak demographic label (int 0-12) dari nama folder video.
 
        Format yang didukung:
          Real : 001_BAL_M_144_REAL   -> parts[1] = 'BAL' langsung
          VC   : 001_BAL_M_144_VC     -> parts[1] = 'BAL' langsung
          FS   : 001_009_144_FS       -> lookup parts[0] (target ID)
          MM   : 001_009_144_FS+VC+LS -> lookup parts[0] (target ID)
 
        Konvensi dataset: suku yang dipakai = suku TARGET (parts[0]).
 
        Return: int (0-12), default 0 jika tidak ditemukan.
        """
        if 'Indo' not in file_path:
            return 13
 
        parts = video_name.split('_')
        if len(parts) < 1:
            return 13
 
        # Kasus 1: parts[1] adalah kode suku langsung (real & VC)
        if len(parts) >= 2 and parts[1] in SUKU_DICT:
            return SUKU_DICT[parts[1]]
 
        # Kasus 2: parts[1] adalah ID source (FS & MM)
        # Suku diambil dari TARGET ID = parts[0]
        target_id = parts[0]
        if target_id in self.indo_demographics:
            return self.indo_demographics[target_id]
 
        # Fallback: coba source ID = parts[1]
        if len(parts) >= 2 and parts[1] in self.indo_demographics:
            return self.indo_demographics[parts[1]]
 
        return 13
            
    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks


    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        
        # Record video name for video-level metrics
        video_name_list = []

        # Try to get the dataset information from the JSON file
        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace('/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public')
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        target_compressions = self.config.get('compressions', ['c23', 'c40', 'raw', '720p', '144p'])

        NO_COMPRESSION_DATASETS = [
            'Celeb-DF-v2', 'Celeb-DF-v1', 'DFDC', 'DFDCP', 
            'DeeperForensics-1.0', 'UADFV'
        ]
        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            mode_data = dataset_info[dataset_name][label][self.mode]
            
            if dataset_name in NO_COMPRESSION_DATASETS:
                # Struktur JSON-nya langsung: mode_data = {video_name: video_info}
                # Tidak ada compression level, bungkus langsung
                compression_items = [('no_comp', mode_data)]
            else:
                # Struktur JSON-nya: mode_data = {comp_level: {video_name: video_info}}
                compression_items = [
                    (comp, data) for comp, data in mode_data.items()
                    if comp in target_compressions
                ]
            # Looping untuk mencari folder kompresi (c23, c40, 720p, 144p) di dalam JSON
            for comp_level, sub_dataset_info in compression_items:

                # Iterate over the videos in the dataset
                for video_name, video_info in sub_dataset_info.items():
                    # Unique video name (tambahan nama kompresi agar tidak bentrok)
                    unique_video_name = f"{video_info['label']}_{comp_level}_{video_name}"

                    if video_info['label'] not in self.config['label_dict']:
                        raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                    
                    label_idx = self.config['label_dict'][video_info['label']]
                    frame_paths = video_info['frames']
                    
                    # sorted video path to the lists
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                    else:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                        
                    total_frames = len(frame_paths)
                    if self.frame_num < total_frames:
                        total_frames = self.frame_num
                        if self.video_level:
                            # Select clip_size continuous frames
                            start_frame = random.randint(0, total_frames - self.frame_num) if self.mode == 'train' else 0
                            frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
                        else:
                            # Select self.frame_num frames evenly distributed throughout the video
                            step = total_frames // self.frame_num
                            frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                
                    # If video-level methods, crop clips from the selected frames if needed
                    if self.video_level:
                        if self.clip_size is None:
                            raise ValueError('clip_size must be specified when video_level is True.')
                        # Check if the number of total frames is greater than or equal to clip_size
                        if total_frames >= self.clip_size:
                            # Initialize an empty list to store the selected continuous frames
                            selected_clips = []
                            num_clips = total_frames // self.clip_size
                            if num_clips > 1:
                                clip_step = (total_frames - self.clip_size) // (num_clips - 1)
                                for i in range(num_clips):
                                    start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1)) if self.mode == 'train' else i * clip_step
                                    continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                    selected_clips.append(continuous_frames)
                            else:
                                start_frame = random.randrange(0, total_frames - self.clip_size + 1) if self.mode == 'train' else 0
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                selected_clips.append(continuous_frames)

                            label_list.extend([label_idx] * len(selected_clips))
                            frame_path_list.extend(selected_clips)
                            video_name_list.extend([unique_video_name] * len(selected_clips))

                        else:
                            selected_clips = []
                            continuous_frames = frame_paths.copy()
                            while len(continuous_frames) < self.clip_size:
                                continuous_frames.append(continuous_frames[-1])
                            
                            selected_clips.append(continuous_frames)
                            label_list.extend([label_idx] * len(selected_clips))
                            frame_path_list.extend(selected_clips)
                            video_name_list.extend([unique_video_name] * len(selected_clips))
                    
                    # Otherwise, extend the label and frame paths to the lists according to the number of frames
                    else:
                        # Extend the label and frame paths to the lists according to the number of frames
                        label_list.extend([label_idx] * total_frames)
                        frame_path_list.extend(frame_paths)
                        # video name save
                        video_name_list.extend([unique_video_name] * len(frame_paths))
            
        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        
        return frame_path_list, label_list, video_name_list

     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution'] # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            if not file_path.startswith('.') and not file_path.startswith('/'):
                file_path = os.path.join(self.config["rgb_dir"], file_path)
            
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                clean_path = file_path
                marker = "datasets/rgb/"
                if marker in clean_path:
                    clean_path = clean_path[clean_path.find(marker) + len(marker):]
                elif clean_path.startswith('./datasets\\'):
                    clean_path = clean_path.replace('./datasets\\', '')
                elif clean_path.startswith('./datasets/'):
                    clean_path = clean_path.replace('./datasets/', '')
                
                # Standarkan semua slash jadi forward slash
                clean_path = clean_path.replace('\\', '/')
                # Ganti hanya slash pertama jadi backslash (sesuai format dataset2lmdb)
                lmdb_key = clean_path.replace('/', '\\', 1)
                # -------------------------------

                image_bin = txn.get(lmdb_key.encode('utf-8'))
                if image_bin is None:
                    print(f"\n[FATAL ERROR] GAMBAR TIDAK DITEMUKAN DI LMDB!")
                    print(f"Path dari JSON  : {file_path}")
                    print(f"Key yg dicari   : {lmdb_key}")
                    import sys; sys.exit(1)
                
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))


    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size,3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path.startswith('.') and not file_path.startswith('/'):
                file_path = os.path.join(self.config["rgb_dir"], file_path)
                
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark=self.rescale_landmarks(np.float32(landmark), original_size=256, new_size=self.config['resolution'])
        return landmark

    def load_audio(self, image_path):
        """Memuat file .wav dari direktori audios dan mengonversi ke Mel-Spectrogram."""
        folder_video_path = os.path.dirname(image_path)
        video_name = os.path.basename(folder_video_path)
        base_dir = os.path.dirname(folder_video_path).replace('frames', 'audios')
        audio_path = os.path.join(base_dir, f"{video_name}.wav")

        n_mels = self.config.get('audio_config', {}).get('n_mels', 128)
        target_length = self.config.get('audio_config', {}).get('target_length', 256)
        sample_rate = self.config.get('audio_config', {}).get('sample_rate', 16000)

        # Jika audio gagal diekstrak / tidak ada, kembalikan tensor nol
        if not os.path.exists(audio_path):
            return torch.zeros((1, n_mels, target_length))

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Konversi ke Mel-Spectrogram
            mel_spec = TA_transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512
            )(waveform)
            # Ubah ke skala Logaritma (Decibel) agar polanya lebih jelas
            mel_spec = TA_transforms.AmplitudeToDB()(mel_spec)

            # Samakan panjang waktu (padding/truncating) agar bisa di-batch
            if mel_spec.shape[-1] > target_length:
                mel_spec = mel_spec[..., :target_length]
            else:
                pad_amount = target_length - mel_spec.shape[-1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
            return mel_spec # Output: [1, 128, 256]
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros((1, n_mels, target_length))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_paths = [p.replace('\\', '/') for p in image_paths]
        
        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
            

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)


        first_img_path = image_paths[0] if isinstance(image_paths, list) else image_paths
        if '\\' in first_img_path:
            video_name = first_img_path.split('\\')[-2]
        else:
            video_name = first_img_path.split('/')[-2]
        
        # Default label jika format tidak sesuai (misal untuk dataset FF++)
        demographic_label = self._get_demographic_label(video_name, first_img_path)
 
        if self.video_level:
            image_tensors = torch.stack(image_tensors, dim=0)
            if not any(lm is None or (isinstance(lm, list) and None in lm) for lm in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            image_tensors = image_tensors[0]
            if not any(lm is None or (isinstance(lm, list) and None in lm) for lm in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]
 
        audio_tensor = self.load_audio(first_img_path) if self.config.get('with_audio', False) else None
 
        return image_tensors, label, landmark_tensors, mask_tensors, demographic_label, audio_tensor

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks, demographics, audios = zip(*batch)
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        demographics = torch.LongTensor(demographics)
        if not any(a is None for a in audios):
            audios = torch.stack(audios, dim=0) # [B, 1, 128, 256]
        else:
            audios = None

        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        data_dict['demographic_label'] = demographics
        data_dict['audio'] = audios
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config = config,
                mode = 'train', 
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
