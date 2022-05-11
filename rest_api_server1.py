from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from multiprocessing.pool import ThreadPool
from queue import Queue

import base64
from skimage import measure
from collections import deque
from typing import List
from sklearn.utils.linear_assignment_ import linear_assignment
from tracking.kalman_tracker import KalmanTracker
from tracking import UnitObject
from tracking.base_tracker import BaseTracker

net = Yolact()
hostname = 'localhost'
PORT = 6337


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons 

class bbox():
    def __init__(self, x, y, w, h, obj_id, track_id):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.obj_id = obj_id
        self.track_id = track_id

class DetectAndTrack:
    """
    Class that connects detection and tracking
    """

    def __init__(self, detector, tracker):
        self.max_age = 4
        self.min_hits = 1
        self.frame_count = 0
        self.tracker_list: List[BaseTracker] = []
        self.track_id_list = deque(list(map(str, range(100))))
        self.detector = detector
        self.tracker = tracker

    def pipeline(self, img):
        """
        Pipeline to process detections and trackers
        :param img: current frame
        :return: frame with annotation
        """

        self.frame_count += 1

        unit_detections = self.detector.get_detections(img)  # measurement

        unit_trackers = []

        for trk in self.tracker_list:
            unit_trackers.append(trk.unit_object)

        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(unit_trackers, unit_detections,
                                                                                     iou_thrd=0.3)

        # Matched Detections
        for trk_idx, det_idx in matched:
            z = unit_detections[det_idx].box
            z = np.expand_dims(z, axis=0).T
            tmp_trk = self.tracker_list[trk_idx]
            tmp_trk.predict_and_update(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            unit_trackers[trk_idx].box = xx
            unit_trackers[trk_idx].class_id = unit_detections[det_idx].class_id
            tmp_trk.unit_object = unit_trackers[trk_idx]
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

        # Unmatched Detections
        for idx in unmatched_dets:
            z = unit_detections[idx].box
            z = np.expand_dims(z, axis=0).T
            tmp_trk = self.tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.unit_object.box = xx
            tmp_trk.unit_object.class_id = unit_detections[idx].class_id
            tmp_trk.tracking_id = self.track_id_list.popleft()  # assign an ID for the tracker
            self.tracker_list.append(tmp_trk)
            unit_trackers.append(tmp_trk.unit_object)

        # Unmatched trackers
        for trk_idx in unmatched_trks:
            tmp_trk = self.tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.unit_object.box = xx
            unit_trackers[trk_idx] = tmp_trk.unit_object

        # The list of tracks to be annotated
        good_tracker_list = []
        for trk in self.tracker_list:
            if (trk.hits >= self.min_hits) and (trk.no_losses <= self.max_age):
                good_tracker_list.append(trk)
                # img = utils.drawing.draw_box_label(img, trk, self.detector.class_names)

        # Manage Tracks to be deleted
        deleted_tracks = filter(lambda x: x.no_losses > self.max_age, self.tracker_list)

        for trk in deleted_tracks:
            self.track_id_list.append(trk.tracking_id)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses <= self.max_age]
        return good_tracker_list
        # return img

    @staticmethod
    def assign_detections_to_trackers(unit_trackers: List[UnitObject], unit_detections: List[UnitObject], iou_thrd=0.3):
        """
        Matches Trackers and Detections
        :param unit_trackers: trackers
        :param unit_detections: detections
        :param iou_thrd: threshold to qualify as a match
        :return: matches, unmatched_detections, unmatched_trackers
        """
        IOU_mat = np.zeros((len(unit_trackers), len(unit_detections)), dtype=np.float32)
        for t, trk in enumerate(unit_trackers):
            for d, det in enumerate(unit_detections):
                if trk.class_id == det.class_id:
                    IOU_mat[t, d] = utils.box_utils.calculate_iou(trk.box, det.box)

        # Finding Matches using Hungarian Algorithm
        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t, trk in enumerate(unit_trackers):
            if t not in matched_idx[:, 0]:
                unmatched_trackers.append(t)

        for d, det in enumerate(unit_detections):
            if d not in matched_idx[:, 1]:
                unmatched_detections.append(d)

        matches = []

        # Checking quality of matched by comparing with threshold
        for m in matched_idx:
            if IOU_mat[m[0], m[1]] < iou_thrd:
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def tracking_id(cur_bbox_vec, change_history, frames_story, max_dist):
    prev_track_id_present = False

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=1, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(
            dets_out, w, h,
            visualize_lincomb=args.display_lincomb,
            crop_masks=args.crop,
            score_threshold=args.score_threshold
        )
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]

        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        masks_np = masks.cpu().numpy()
        print(masks_np.shape)
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
    ###
    print(boxes)
    masks_np = masks.view(-1, h, w).cpu().numpy()
    print(masks_np.shape)
    masks_np_0 = masks_np[0,:,:] #* 255
    

    for i in range(0, len(classes)):
        instance_mask = masks_np[i, :, :]
        print("polygons:", binary_mask_to_polygon(instance_mask, 4))
    ## debugg
    # cv2.imwrite("mask0.png", masks_np_0)

    # contours = measure.find_contours(masks_np_0, 0.5)
    # segmentations = []
    # for contour in contours:
    #     contour = np.flip(contour, axis=1)
    #     segmentation = contour.ravel().tolist()
    #     segmentations.append(segmentation)
    # print(masks_np_0.shape)
    # rle = pycocotools.mask.encode(np.asfortranarray(masks_np_0.astype(np.uint8)))
    # area = pycocotools.mask.area(rle)

    # print("segmentations:", segmentations)
    # print("polygons:", binary_mask_to_polygon(masks_np_0, 4))
    # print("classes:", classes, scores)

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]

        masks_color_summand = masks_color[0]
        """
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        """
        # img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        img_gpu = masks_color_summand    # testing by nick


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        return img_numpy

    
    return img_numpy


def get_result_meta(dets_out, img, h, w, undo_transform=True):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(
            dets_out, w, h,
            visualize_lincomb=args.display_lincomb,
            crop_masks=args.crop,
            score_threshold=args.score_threshold
        )
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]

        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        masks_np = masks.cpu().numpy()
        print(masks_np.shape)
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break
   
    masks_np = masks.view(-1, h, w).cpu().numpy()

    instances = list()
    for i in range(0, len(classes)):
        instance_mask = masks_np[i, :, :]
        # print("polygons:", binary_mask_to_polygon(instance_mask, 4))
        instance = dict()
        instance['class'] = int(classes[i])
        instance['bbox'] = {
            'x': int(boxes[i][0]),
            'y': int(boxes[i][1]),
            'w': int(boxes[i][2]) - int(boxes[i][0]),
            'h': int(boxes[i][3]) - int(boxes[i][1])
        }
        instance['polygons'] = binary_mask_to_polygon(instance_mask, 2)
        instances.append(instance)
    
    return instances




def evalimage(net:Yolact, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    preds = net(batch)

    # instances = prep_display(preds, frame, None, None, undo_transform=False)
    # ret, jpg_buffer = cv2.imencode(".jpg", img_numpy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # encoded_string = base64.b64encode(jpg_buffer)
    instances = get_result_meta(preds, frame, None, None, undo_transform=False)

    return instances

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


def init_engine(model=None):
    parse_args()

    if model:
        args.trained_model = f'weights/{model}'
    else:
        args.trained_model = f'weights/yolact_resnet50_54_800000.pth'
        # args.trained_model = f'weights/yolact_base_54_800000.pth'

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    print(args.trained_model)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        global net
        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()


def process_image(image_path, top_k, score_threshold):
    global args

    args.image = image_path
    args.top_k = top_k
    args.score_threshold = score_threshold

    with torch.no_grad():
        net.detect.use_fast_nms = args.fast_nms
        net.detect.use_cross_class_nms = args.cross_class_nms
        cfg.mask_proto_debug = args.mask_proto_debug

        # TODO Currently we do not support Fast Mask Re-scoring in evalimage
        detected_objects = evalimage(net, args.image)

    return detected_objects


class HttpServerHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.startswith('/init_engine'):
            try:
                full_path = f'http://{hostname}:{PORT}{self.path}'
                parsed_url = urlparse(full_path)
                if 'model' in parse_qs(parsed_url.query):
                    model = parse_qs(parsed_url.query)['model'][0]
                    print(f'Model: {model}')
                    init_engine(model)
                    self.send_successs_response('Loading model... Done.')
                    return
                else:
                    self.send_bad_request_response("Could not find 'model' from your request.")
                    return
            except:
                self.send_bad_request_response()
                return

        if self.path.startswith('/detect_objects'):
            try:
                full_path = f'http://{hostname}:{PORT}{self.path}'
                parsed_url = urlparse(full_path)
                if 'image_path' in parse_qs(parsed_url.query) and len(parse_qs(parsed_url.query)['image_path']) > 0:
                    image_path = f"{parse_qs(parsed_url.query)['image_path'][0]}"
                else:
                    self.send_bad_request_response("Could not find 'image_path' from your request.")
                    return

                top_k = 20
                try:
                    if 'top_k' in parse_qs(parsed_url.query) and len(parse_qs(parsed_url.query)['top_k']) > 0:
                        top_k = int(parse_qs(parsed_url.query)['top_k'][0])
                except:
                    self.send_bad_request_response("Wrong type value found. 'top_k' must be an integer.")
                    return

                score_threshold = 0.25
                try:

                    if 'score_threshold' in parse_qs(parsed_url.query) and \
                            len(parse_qs(parsed_url.query)['score_threshold']) > 0:
                        score_threshold = float(parse_qs(parsed_url.query)['score_threshold'][0])
                except:
                    self.send_bad_request_response("Wrong type value found. 'score_threshold' must be a float.")
                    return

                st = datetime.now()
                base64_img = process_image(image_path, top_k, score_threshold)
                print(f'Detection time: {datetime.now() - st}')
                self.send_successs_response(base64_img)
                return

            except Exception as e:
                self.send_bad_request_response(str(e))
                return

        self.send_bad_request_response()

    def send_successs_response(self, data):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(data), "utf-8"))

    def send_objects(self, objects):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        data = {
            "LBObjectsCount": len(objects),
            "LBObjects": objects
        }
        self.wfile.write(bytes(json.dumps(data), "utf-8"))

    def send_bad_request_response(self, message=None):
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        if message:
            self.wfile.write(bytes(f"Bad Request: {message}", "utf-8"))
        else:
            self.wfile.write(bytes(f"Bad Request", "utf-8"))


if __name__ == '__main__':
    init_engine()

    webServer = HTTPServer((hostname, PORT), HttpServerHandler)
    print("Server started http://%s:%s" % (hostname, PORT))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
