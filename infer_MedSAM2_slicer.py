from glob import glob
from tqdm import tqdm
import os
from os.path import join, isfile, basename
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime
from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2
import yaml

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def resize_rgb(array, image_size):
    # 입력데이터 크기가져오는데 슬라이스 개수인 d만 가져옴
    d, h, w = array.shape[:3]
    # resize할 크기의 np.zero만들고
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    # 슬라이스 하나씩 돌면서 이미지 resize
    for i in range(d):
        img_rgb = Image.fromarray(array[i].astype(np.uint8))
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

# 입력이 grayscale일 경우 rgb로 바꿔주는 함수
def grayscale2rgb(array):
    if len(array.shape) > 3:
        return array

    rgb_array = np.zeros((*array.shape, 3))
    for i in range(array.shape[0]):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        rgb_array[i] = np.array(img_rgb)

    return rgb_array


@torch.inference_mode()
def infer_3d(predictor, img_npz_file, gts_file, propagate, model_cfg, pred_save_dir):
    print(f'infering {img_npz_file}')
    npz_name = basename(img_npz_file) # input한 파일이름
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # input데이터로 읽은 데이터 변수에 저장 
    # npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # input데이터로 읽은 데이터 변수에 저장 
    boxes_3D = npz_data['boxes']  # (D, num_boxes, 4) -> 사용자가 설정한 경계박스, num_boxes 사용자가 만든 roi개수, 4의 의미는 박스는 점4개니까
    gts = np.load(gts_file, 'r', allow_pickle=True)['segs'] if gts_file != 'X' else None
    # 실제 데이터 = img_3D -> (D, H, W)
    img_3D = npz_data['imgs'] 
    print(f"infer_3d에서 원본 이미지크기: {img_3D.shape}")
    # 픽셀크기 256보다크면 정규화
    if np.max(img_3D) >= 256:
        img_3D = (img_3D - np.min(img_3D)) / (np.max(img_3D) - np.min(img_3D)) * 255
        img_3D = img_3D.astype(np.int16)
    # assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    # grayscale을 rgb로 바꿔주고
    img_3D = grayscale2rgb(img_3D)
    D, H, W = img_3D.shape[:3]
    segs_3D = np.zeros(img_3D.shape[:3], dtype=np.uint8) # segs_3D = segmentation결과 저장용 변수, 원본 iamge랑 크기같음.(segs_3D: (49, 512, 512))
    boxes_3D = npz_data['boxes']  # (D, num_boxes, 4) -> 사용자가 설정한 경계박스, num_boxes 사용자가 만든 roi개수, 4의 의미는 박스는 점4개니까



    #이거이해안감
    z_range = npz_data['z_range'] # (z_min, z_max, slice_idx) -> 현재 사용자가 선택한 슬라이스
    # 원본이미지의 높이와 너비저장(복원할떄 사용)
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]

    with open(join('sam2', model_cfg), 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        image_size = yaml_data['model']['image_size']
    
    # 이미지 resize
    img_resized = resize_rgb(img_3D, image_size)
    img_resized = img_resized / 255.0
    # 딥러닝 모델에 넘겨주어야하니 cpu에있는 넘파이배열을 gpu텐서로
    img_resized = torch.from_numpy(img_resized).cuda()
    # medsam2가 학습했던 데이터셋의 모든 이미지의 평균과 표준편차로, input을 이분포로 맞춰주기 위해서 사용 
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    # input이미지를 평균과 표준편차로 정규화
    img_resized -= img_mean
    img_resized /= img_std
    z_mids = []
    
    z_indices, slice_idx = z_range[:2], z_range[2]
    
    labels = np.unique(gts)
    
    box_num = boxes_3D.shape[1]

    total_z_min = 0
    total_z_max = 0

    for idx, label in enumerate(labels, start=1):
    # for idx in range(1,box_num+1): # (D, num_boxes, 4)
        # gt가 None이라 gt에는 전부0
        gt = (gts == (label))
        # gt에서 True인 좌표를 모두 찾아서 indicies에 할당
        indices = np.where(gt)

        print(f"인디시스: {indices}")

        # 그리고 좌표의 첫번째 인덱스의 z좌표를 z_mid_orig에 저장
        z_mid_orig = np.quantile(indices[0], 0.5).astype(int)
        
        print(f"z_mid_orig : {z_mid_orig}")


        # z축의 최소최대를 찾는데, 이거 이전에 분할한 객체들이 존재하는 z만 찾아서 해당 슬라이스에서면 전파하려고하는듯
        z_min = z_indices.min() if z_indices.size > 0 else None
        z_max = z_indices.max() if z_indices.size > 0 else None

        total_z_min = z_min if z_min < total_z_min else 0
        total_z_max = z_max if z_max > total_z_max else 0

        # 전체 이미지에서 해당하는 이미지만 선택 = img
        img = img_resized[z_min:(z_max+1)]
        z_mid = int(img.shape[0]/2)
        z_mids.append(z_mid_orig)
        mask_prompt = gt[z_mid_orig] # 특정 z에서의 segmentatino된 2D 슬라이스
        #  잘라낸 이미지 내에서 힌트 마스크의 상대적인 위치를 계산(원래는 전체 슬라이스에서의 상대적위치였으니)
        ann_frame_idx = z_mid_orig - (z_min if z_min is not None else 0)

        print('analyzed image size', img.shape, 'mid idx', z_mid) # ([37, 3, 512, 512]) mid idx 18
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # input img is shape depth_to_consider, 3, 512, 512
            # 예측기 init
            inference_state = predictor.init_state(img, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=ann_frame_idx, obj_id=1, mask=mask_prompt)
            # 분할결과인 masks를 segs_3D에 저장
            segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = label
            # run propagation throughout the video and collect the results in a dict
            #video_segments = {}  # video_segments contains the per-frame segmentation results
            # 정방향전파
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = label
            predictor.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=ann_frame_idx, obj_id=1, mask=mask_prompt)
            # 역방향전파
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = label
            predictor.reset_state(inference_state)
        
        # z축 최소 최대를 찾아서 원본 크기랑 비교해서 나머지 0으로 채워주기( 그럼 라벨이 여러개일때 문제 생기는데 이거 해결해야됨)
    # 총 깊이 = D
    padding_iwdth = ((total_z_min, D-total_z_max),(0,0),(0,0))
    full_segs_3D = np.pad(segs_3D, pad_width=padding_iwdth, mode='constant', constant_values=0)
    
    


    print(np.unique(full_segs_3D))
    np.savez_compressed(join(pred_save_dir, npz_name), segs=full_segs_3D)

    return inference_state


@torch.inference_mode()
def improve_3d(predictor, inference_state, img_npz_file, pred_save_dir):
    # input이미지 크기가져와서 빈 넘파이배열하나 만들어주기(분할결과저장용)
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    segs_3D = np.zeros(npz_data['img_size'], dtype=np.uint8)
    # 
    points_addition = npz_data['points_addition']
    points_subtraction = npz_data['points_subtraction']
    labels = np.array([1]*points_addition.shape[0] + [0]*points_subtraction.shape[0], dtype=np.int32)
    if points_addition.shape[0] == 0:
        points = points_subtraction
    elif points_subtraction.shape[0] == 0:
        points = points_addition
    else:
        points = np.vstack((points_addition, points_subtraction))
    print(labels, points, sep='\n')
    # pass points, labels, and the (box, zrange, slice_idx)
    box = np.array(npz_data['bboxes'][0], np.float32)
    z_min = min(npz_data['zrange']) # 객체가있는 최소 z
    z_mid_orig = int(points[0, -1]) # 사용자가 분할중인 z좌표
    ann_frame_idx = z_mid_orig - z_min # 상대위치로변환(전체슬라이스에서의 인덱스로변환)

    points = points[:,:-1].astype(np.float32) # dropping 3rd dimension

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 사용자 수정을 반영해서 마스크 추론
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        # segs_3D 에 저장
        segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = 1
        # run propagation throughout the video and collect the results in a dict
        #video_segments = {}  # video_segments contains the per-frame segmentation results
        # 수정된 마스크기반 양방향전파
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            print(out_frame_idx)
            segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            print(out_frame_idx)
            segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)

    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

    return inference_state



def perform_inference(checkpoint, cfg, img_path, gts_path, propagate, pred_save_dir):
    # img_path = 원본이미지와 경계박스 정보들을 파일 저장해놓은곳

    # make propagate boolean
    predictor = build_sam2_video_predictor_npz(cfg, checkpoint) if propagate else SAM2ImagePredictor(build_sam2(cfg, checkpoint, device="cuda"))

    os.makedirs(pred_save_dir, exist_ok=True)

    inference_state = infer_3d(predictor, img_path, gts_path, propagate, cfg, pred_save_dir)

    return predictor, inference_state

def improve_inference(img_path, pred_save_dir, predictor_state):
    # make propagate boolean
    predictor, inference_state = predictor_state['predictor'], predictor_state['inference_state']

    os.makedirs(pred_save_dir, exist_ok=True)

    inference_state = improve_3d(predictor, inference_state, img_path, pred_save_dir)

    return predictor, inference_state



if __name__ == '__main__':
    perform_inference('checkpoints/MedSAM2_latest.pt', 'MedSAM2_tiny512.yaml', 'img_data.npz', 'X', False, 'data/video/segs_tiny')
    print('Server is installed!')
