import tempfile
from os import path as osp
import os
import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from .utils import nuscenes_get_rt_matrix, get_sensor_transforms, downsample_intrinsic
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from tqdm import tqdm
import math
import torch


from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes

@DATASETS.register_module()
class NuScenesOccDataset(Custom3DDataset):
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file=None,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 test_mode=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                 ego_cam='CAM_FRONT',
                 # SOLLOFusion
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 aux_frames=[1],
                 gaussian_label_path=None,
                 save_sample=True,
                 eval_miou=False,
                 load_2d_label=False,
                 selected_scene=False,
                ):
        self.load_interval = load_interval
        self.selected_scene = selected_scene
        print('load_interval: ', self.load_interval)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode)
        self.occupancy_path = occupancy_path

        if self.modality is None:
            self.modality = dict(
                    use_lidar=False,
                    use_camera=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False)


        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam

        # SOLOFusion
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        if self.test_mode:
            assert self.sequences_split_num == 1
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

        #3D Gaussians splatting
        self.cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK',
                          'CAM_BACK_RIGHT']
        self.aux_frames = aux_frames
        self.gaussian_label_path = gaussian_label_path
        self.save_sample = save_sample
        self.eval_miou = eval_miou
        self.load_2d_label = load_2d_label


    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        if self.selected_scene:
            data_infos = [item for item in data_infos if item.get('scene_name') == self.selected_scene]

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos


    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
           
        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['prev']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_gaussian_label(self, index, img_size=[900, 1600], ds_rate=4):
        img_size = np.array(img_size) // ds_rate
        info = self.data_infos[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            if select_id >= len(self.data_infos) or self.data_infos[select_id]['scene_name'] != info['scene_name']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]

            for cam_name in self.cam_names:
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)


                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)

                time_ids[time_id].append(idx)
                idx += 1

        T, N = len(self.aux_frames) + 1, len(info['cams'].keys())

        intrins = torch.stack(intrins)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)

        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)
        intrins = intrins.view(T, N, 3, 3)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :, ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        intrins = downsample_intrinsic(intrins, downsampling_factor=ds_rate)

        sample_mask = torch.zeros(200, 200, 16)

        return (sensor2keyegos, intrins, sample_mask)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            index=index,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            lidarseg_filename=info.get('lidarseg_filename', 'None') 
        )
        # if 'ann_infos' in info:
        #     input_dict['ann_infos'] = info['ann_infos']

        if self.gaussian_label_path is not None:
            input_dict['gaussian_labels'] = self.get_gaussian_label(index)

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                cam_positions = []

                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)
                    cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
                    cam_positions.append(cam_position.flatten()[:3])
                   

                input_dict.update(
                    dict(
                        
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                # if not self.test_mode:
                #     annos = self.get_ann_info(index)
                #     input_dict['ann_info'] = annos
            else:   
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
            if self.use_sequence_group_flag:
                input_dict['sample_index'] = index
                input_dict['sequence_group_idx'] = self.flag[index]
                input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]
                # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
                # if they belong to same sequence.
                input_dict['nuscenes_get_rt_matrix'] = dict(
                    lidar2ego_rotation = self.data_infos[index]['lidar2ego_rotation'],
                    lidar2ego_translation = self.data_infos[index]['lidar2ego_translation'],
                    ego2global_rotation = self.data_infos[index]['ego2global_rotation'],
                    ego2global_translation = self.data_infos[index]['ego2global_translation'],
                )
                if not input_dict['start_of_sequence']:
                    input_dict['curr_to_prev_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index - 1],
                        "lidar", "lidar"))
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index - 1], self.data_infos[index],
                        "lidar", "global")) # TODO: Note that global is same for all.
                    input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index - 1],
                        "ego", "ego"))
                else:
                    input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix( 
                        self.data_infos[index], self.data_infos[index], "lidar", "global")
                        )
                    input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index],
                        "ego", "ego"))
                input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index],
                    "global", "lidar"))
        
        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        for select_id in range(*self.multi_adj_frame_id_cfg):
            if select_id == 0: continue
            select_id = min(max(index - select_id, 0), len(self.data_infos)-1)

            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list


    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES
       
        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_id = det.get('index', sample_id)
            # from IPython import embed
            # embed()
            # exit()

            sample_token = self.data_infos[sample_id]['token']

            
            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                pass
                # nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def evaluate(self, results,
                       logger=None,
                        metric='bbox',
                        jsonfile_prefix='test',
                        result_names=['pts_bbox'],
                        show=False,
                        out_dir=None,
                        pipeline=None,
                        save=False,
                        ):
            results_dict = {}
            
            if results[0].get('pred_occupancy', None) is not None:
                if self.eval_miou:
                    results_dict.update(self.evaluate_occupancy_miou(results, show_dir=jsonfile_prefix, save=save))
                else:
                    results_dict.update(self.evaluate_occupancy(results, show_dir=jsonfile_prefix, save=save))
            
            print('done')
              

    def evaluate_occupancy(self, occ_results, runner=None, show_dir='./show', save=False, **eval_kwargs):
        print(show_dir)
        print('\nStarting Evaluation...')
        processed_set = set()
        results_dict = {}
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            occ_pred = occ_pred_w_index['pred_occupancy']
            flow = occ_pred_w_index['pred_flow']

            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_token = info['token']

            if show_dir is not None:
                if self.save_sample:
                    save_item = {}
                    save_item['semantics'] = occ_pred
                    save_item['flow'] = flow if flow is not None else np.zeros((200, 200, 16, 2))
                    if 'pred_opacity' in occ_pred_w_index:
                        save_item['opacity'] = occ_pred_w_index['pred_opacity']
                    if ('semantics_2d' in occ_pred_w_index) & ('depths_2d' in occ_pred_w_index):
                        save_item['semantics_2d'] = occ_pred_w_index['semantics_2d']
                        save_item['depths_2d'] = occ_pred_w_index['depths_2d']
                        #save_item['flow'] =  np.zeros((200, 200, 16, 2))
                    #save_item['flow'] = flow
                    save_path = os.path.join(show_dir, scene_name, sample_token, 'preds.npz')
                    os.makedirs(os.path.join(show_dir, scene_name, sample_token), exist_ok=True)
                    #np.save(save_path, save_item)
                    np.savez_compressed(save_path, **save_item)
                else:
                    save_item = {}
                    save_item['semantics'] = occ_pred
                    save_item['flow'] = flow if flow is not None else np.zeros((200, 200, 16, 2))
                    results_dict[sample_token] = save_item
        print('complete')
        if not self.save_sample and show_dir is not None:
            from ray_iou.ray import quick_save
            quick_save(results_dict=results_dict, save_dir=show_dir)

        print('done')
        exit()

    def evaluate_occupancy_miou(self, occ_results, runner=None, show_dir=None, save=False, **eval_kwargs):
        from .occ_metrics import Metric_mIoU, Metric_FScore, Metric_mIoU_SurroundOcc
        if show_dir is not None:
            # import os
            # if not os.path.exists(show_dir):

            mmcv.mkdir_or_exist(show_dir)
            mmcv.mkdir_or_exist(os.path.join(show_dir, 'occupancy_pred'))
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin = 0  # eval_kwargs.get('begin',None)

            end = 1 if not save else len(occ_results)  # eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU_SurroundOcc(
            num_classes=17,
            use_lidar_mask=False,
            use_image_mask=True)

        self.eval_fscore = False

        count = 0
        print('\nStarting Evaluation...')
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            occ_pred = occ_pred_w_index['pred_occupancy']
            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_token = info['token']
            occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
            occ_gt = np.load(occupancy_file_path)

            gt_semantics = occ_gt['semantics']

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics)

        res = self.occ_eval_metrics.count_miou()

        return res


    def evaluate_occupancy2(self, occ_results, runner=None, show_dir=None, save=False, **eval_kwargs):
        from .occ_metrics import Metric_mIoU, Metric_FScore
        if show_dir is not None:
            # import os
            # if not os.path.exists(show_dir):

            mmcv.mkdir_or_exist(show_dir)
            mmcv.mkdir_or_exist(os.path.join(show_dir, 'occupancy_pred'))
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin= 0 # eval_kwargs.get('begin',None)

            end=1 if not save else len(occ_results) # eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        
        self.eval_fscore = False
        if  self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
        count = 0
        print('\nStarting Evaluation...')
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            occ_pred = occ_pred_w_index['pred_occupancy']
            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_token = info['token']
            occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
            occ_gt = np.load(occupancy_file_path)
 
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)            
            if show_dir is not None:
                if begin is not None and end is not None:
                    if index>= begin and index<end:
                        sample_token = info['token']
                        count += 1
                        save_path = os.path.join(show_dir, 'occupancy_pred', scene_name+'_'+sample_token)
                        np.savez_compressed(save_path, pred=occ_pred[mask_camera], gt=occ_gt, sample_token=sample_token)
                        with open(os.path.join(show_dir, 'occupancy_pred', 'file.txt'),'a') as f:
                            f.write(save_path+'\n')
                        np.savez_compressed(save_path+'_gt', pred= occ_gt['semantics'], gt=occ_gt, sample_token=sample_token)
                # else:
                #     sample_token=info['token']
                #     save_path=os.path.join(show_dir,str(index).zfill(4))
                #     np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)


            self.occ_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)
   
        res = self.occ_eval_metrics.count_miou()
        if self.eval_fscore:
            res.update(self.fscore_eval_metrics.count_fscore())

        return res 
        

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

