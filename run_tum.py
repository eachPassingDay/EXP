import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('dg_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import argparse

from dg_slam.dg_model import dg_model
from dg_slam.config import load_config
from dg_slam.gaussian.common import setup_seed
from lietorch import SE3
from evaluation.tartanair_evaluator import TartanAirEvaluator
from dg_slam.gaussian.gaussian_render import render

def parse_list(filepath, skiprows=0):
    """ read list data """
    data = np.loadtxt(filepath, delimiter=' ',
                        dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))
        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))
    return associations

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

def get_tensor_from_camera(RT, Tquad=False):
    """ Convert transformation matrix to quaternion and translation. """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]

    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    quad = rot.as_quat()
    quad = np.roll(quad, 1)

    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)

    tensor[3:] = tensor[3:][[1,2,3,0]]
    return tensor

def loadtum(datapath, frame_rate=-1):
    """ read video data in tum-rgbd format """
    if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
        pose_list = os.path.join(datapath, 'groundtruth.txt')
    elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
        pose_list = os.path.join(datapath, 'pose.txt')

    image_list = os.path.join(datapath, 'rgb.txt')
    depth_list = os.path.join(datapath, 'depth.txt')

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:, 1:].astype(np.float64)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    associations = associate_frames(
        tstamp_image, tstamp_depth, tstamp_pose)

    indicies = [0]
    for i in range(1, len(associations)):
        t0 = tstamp_image[associations[indicies[-1]][0]]
        t1 = tstamp_image[associations[i][0]]
        if t1 - t0 > 1.0 / frame_rate:
            indicies += [i]

    images, poses, depths, seg_masks = [], [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images += [os.path.join(datapath, image_data[i, 1])]
        seg_masks += [os.path.join(datapath, 'seg_mask/' + image_data[i, 1].split('/')[-1])]
        depths += [os.path.join(datapath, depth_data[j, 1])]
        c2w = pose_vecs[k]
        c2w = torch.from_numpy(c2w).float()
        poses += [c2w]

    return images, depths, poses, seg_masks

def as_intrinsics_matrix(intrinsics):
    """ Get matrix representation of intrinsics. """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[535.4, 539.2, 320.1, 247.6], png_depth_scale = 5000.0):
    """ image generator """
    # TUM
    # read all png images in folder
    color_paths, depth_paths, poses, seg_masks = loadtum(datapath, frame_rate=32)

    data = []
    for t in range(len(color_paths)):
        images = [cv2.cvtColor(cv2.resize(cv2.imread(color_paths[t]), (image_size[1], image_size[0])) , cv2.COLOR_BGR2RGB)]
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)
        pose = poses[t]

        seg_mask_data_ori = cv2.imread(seg_masks[t], cv2.IMREAD_GRAYSCALE)
        seg_mask_data_ori = np.where(seg_mask_data_ori != 12, np.zeros_like(seg_mask_data_ori), np.ones_like(seg_mask_data_ori))
        kernel = np.ones((8, 8), np.uint8)
        seg_mask_data = cv2.dilate(seg_mask_data_ori, kernel, iterations=1)
        seg_mask_data = cv2.resize(seg_mask_data, (image_size[1], image_size[0]))
        seg_mask_data = torch.from_numpy(seg_mask_data.astype(np.uint8))

        depth_data = cv2.resize(cv2.imread(depth_paths[t], cv2.IMREAD_UNCHANGED), (image_size[1], image_size[0]))
        depth_data = depth_data.astype(np.float32) / png_depth_scale
        depth_data = torch.from_numpy(depth_data)

        data.append((t, images, depth_data, intrinsics, pose, seg_mask_data))
    return data

if __name__ == '__main__':
    print('Start Running DG-SLAM....')
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="/SSD_DISK/datasets/TUM")
    parser.add_argument("--weights", default="checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    parser.add_argument('--config', default="configs/TUM_RGBD/fr3_walk_xyz.yaml", type=str, help='Path to config file.')

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    from data.utils import tum_split
    if args.id >= 0:
        tum_split = [ tum_split[args.id] ]

    ate_list = []
    for scene in tum_split:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()

        args.config = "configs/TUM_RGBD/" + scene + ".yaml"
        cfg = load_config(args.config, 'configs/dg_slam.yaml')
        setup_seed(cfg["setup_seed"])

        save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        dg_slam = dg_model(cfg, args)
        scenedir = os.path.join(args.datapath, scene)

        intrinsics_vec = [cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']]
        data_reader = image_stream(scenedir, intrinsics_vec = intrinsics_vec)
        cfg["data"]["n_img"] = len(data_reader)

        ## tracking and mapping
        traj = []
        for (tstamp, image, depth, intrinsics, pose, seg_mask) in tqdm(data_reader):
            dg_slam.track(tstamp, image, depth, pose, intrinsics, seg_mask)
            traj.append(pose.cpu().numpy())
        traj_ref = np.array(traj) 

        evaluator = TartanAirEvaluator()        
        N = dg_slam.video.counter.value
        traj_est_key = dg_slam.tracking_mapping.estimate_c2w_list[:N]
        traj_est= []
        for i in range(N):
            pose = get_tensor_from_camera(traj_est_key[i], Tquad = True).cpu()
            traj_est.append(pose)

        traj_est = torch.stack(traj_est, dim=0)       
        dg_slam.video.poses[:N] = lietorch.cat([SE3(traj_est)], 0).inv().data
        traj_est = dg_slam.terminate_woBA(data_reader)

        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scene + '_' + cfg["data"]["exp_name"], save_path = save_path)
        print(results)
        ate_list.append(results["ate_score"])

        # ==========================================================
        # 终极版：动态场景 Masked PSNR 评估 (全装甲防御版)
        # ==========================================================
        import numpy as np 
        
        print(f"开始对 {scene} 进行 Masked 渲染评估 (使用 OneFormer 遮罩)...")
        
        # 重新初始化 dataloader 以遍历序列
        data_reader_eval = image_stream(scenedir, intrinsics_vec=intrinsics_vec)
        
        psnr_list = []
        
        # ==========================================================
        # 终极版：动态场景 Masked PSNR 评估 (最简直接调用版)
        # ==========================================================
        import numpy as np 
        
        print(f"开始对 {scene} 进行 Masked 渲染评估 (使用 OneFormer 遮罩)...")
        
        # 重新初始化 dataloader
        data_reader_eval = image_stream(scenedir, intrinsics_vec=intrinsics_vec)
        psnr_list = []
        
        # 【精准定位】：直接指向 gs_tracking_mapping.py 中的实例！
        mapper = dg_slam.tracking_mapping
        eval_device = mapper.gaussians.get_xyz().device
        
        with torch.no_grad():
            for frame_idx, (tstamp, gt_image, depth, intrinsics, gt_pose, seg_mask) in enumerate(tqdm(data_reader_eval)):
                
                # ------ 1. 位姿解析 (保留防御 TUM Numpy 数组崩溃的逻辑) ------
                if frame_idx < len(traj_est):
                    pose_data = traj_est[frame_idx]
                    
                    if isinstance(pose_data, np.ndarray):
                        pose_data = torch.from_numpy(pose_data).float().to(eval_device)
                    else:
                        pose_data = pose_data.float().to(eval_device)
                        
                    if pose_data.dim() == 1 and pose_data.shape[0] >= 7:
                        t = pose_data[1:4] if pose_data.shape[0] == 8 else pose_data[0:3]
                        q = pose_data[4:8] if pose_data.shape[0] == 8 else pose_data[3:7]
                            
                        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
                        R = torch.zeros((3, 3), device=eval_device)
                        R[0,0] = 1 - 2*qy**2 - 2*qz**2; R[0,1] = 2*qx*qy - 2*qz*qw; R[0,2] = 2*qx*qz + 2*qy*qw
                        R[1,0] = 2*qx*qy + 2*qz*qw; R[1,1] = 1 - 2*qx**2 - 2*qz**2; R[1,2] = 2*qy*qz - 2*qx*qw
                        R[2,0] = 2*qx*qz - 2*qy*qw; R[2,1] = 2*qy*qz + 2*qx*qw; R[2,2] = 1 - 2*qx**2 - 2*qy**2
                        
                        c2w_est = torch.eye(4, device=eval_device)
                        c2w_est[:3, :3] = R
                        c2w_est[:3, 3] = t
                    else:
                        c2w_est = pose_data.to(eval_device)
                else:
                    break 
                
                # ------ 2. 图像转换 ------
                if isinstance(gt_image, np.ndarray):
                    gt_image = torch.from_numpy(gt_image)
                gt_img_tensor = gt_image.to(eval_device).float() / 255.0
                if gt_img_tensor.dim() == 3:
                    if gt_img_tensor.shape[-1] == 3: 
                        gt_img_tensor = gt_img_tensor.permute(2, 0, 1).unsqueeze(0)
                    elif gt_img_tensor.shape[0] == 3: 
                        gt_img_tensor = gt_img_tensor.unsqueeze(0)
                
                # ------ 3. 掩码转换 ------
                if isinstance(seg_mask, np.ndarray):
                    seg_mask = torch.from_numpy(seg_mask)
                static_mask = (seg_mask == 0).to(eval_device)
                static_mask_4d = static_mask.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1) 
                
                # ------ 4. 调用原装渲染器 ------
                # 完全对应你在 gs_tracking_mapping.py 里的写法
                render_pkg = render(
                    mapper.gaussians.get_xyz(), 
                    mapper.gaussians.get_features_dc(), 
                    mapper.gaussians.get_features_rest(),
                    mapper.opacity_activation(mapper.gaussians.get_opacity()), 
                    mapper.scaling_activation(mapper.gaussians.get_scaling()), 
                    mapper.rotation_activation(mapper.gaussians.get_rotation()),
                    mapper.gaussians.get_active_sh_degree(), 
                    mapper.gaussians.get_max_sh_degree(),
                    c2w_est[:3, 3], 
                    torch.inverse(c2w_est).transpose(0, 1), 
                    mapper.projection_matrix, 
                    mapper.fovx, mapper.fovy, mapper.H, mapper.W
                )
                render_img = render_pkg["render"].unsqueeze(0).clamp(0, 1) 
                
                # ------ 5. 计算 Masked PSNR ------
                render_valid = render_img[static_mask_4d]
                gt_valid = gt_img_tensor[static_mask_4d]
                
                if render_valid.numel() > 0:
                    mse = torch.mean((render_valid - gt_valid) ** 2)
                    frame_psnr = -10.0 * torch.log10(mse + 1e-10) 
                    psnr_list.append(frame_psnr.item())


        # 汇总当前场景指标
        scene_avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
        print(f"[{scene}] 平均 Masked PSNR: {scene_avg_psnr:.4f}")
        
        # ================= [新增：输出并保存每一帧的 PSNR] =================
        # 1. 在终端格式化打印出来（保留 4 位小数，方便直接看）
        #formatted_psnr_list = [round(p, 4) for p in psnr_list]
        #print(f"[{scene}] 所有帧的 Masked PSNR 列表: \n{formatted_psnr_list}")
        
        # 2. 自动保存为 txt 文件，方便你后续直接写个 python 脚本画折线图
        psnr_log_path = os.path.join(save_path, f"{scene}_psnr_per_frame.txt")
        with open(psnr_log_path, "w") as f:
            f.write(f"Average Masked PSNR: {scene_avg_psnr:.4f}\n")
            f.write("Frame_Index\tMasked_PSNR\n")
            for idx, p in enumerate(psnr_list):
                f.write(f"{idx}\t{p:.4f}\n")
                
        print(f"已将每帧数据落盘保存至: {psnr_log_path}")
        # ==========================================================
        # ==========================================================
    
    print(ate_list)