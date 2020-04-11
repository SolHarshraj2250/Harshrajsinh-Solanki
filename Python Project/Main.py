#=======================================================================================
#Importing Necessarry or Required APIS or Packages:-
#=======================================================================================
#To read the Video:-
import cv2
#For GUI Generation and For its Work Purpose:-   
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog 
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import  QThread, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
#For Detecting the Objects:-
from config import *
#
from PIL import Image
import copy
#For Importing Other ".py" Files of My Project:-
from detection.detect2 import *
from detection.detect1 import *
#
from __future__ import division
#To Compute Tensor Computing (Multidimensional Array Computing):-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#For Some Array Operation:-
import numpy as np
#For Some Graphical Purpose:-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#=================================================================
#Total type of Objects Which are Part of that Video (Input) With their RGB Coordinates:-
#=================================================================
names = ["bicycle","bus","car","motorbike","truck"]
color_dict = {"bicycle": (179, 52, 255),
              "bus": (255, 191, 0),
              "car": (127, 255, 0),
              "motorbike": (0, 140, 255),
              "truck": (0, 215, 255)}
class CounterThread(QThread):
    sin_counterResult = pyqtSignal(np.ndarray)
    sin_runningFlag = pyqtSignal(int)
    sin_videoList = pyqtSignal(list)
    sin_countArea = pyqtSignal(list)
    sin_done = pyqtSignal(int)
    sin_counter_results = pyqtSignal(list)
    sin_pauseFlag = pyqtSignal(int)
    def __init__(self,model,class_names,device):
        super(CounterThread,self).__init__()
        self.model = model
        self.class_names = class_names
        self.device = device
        self.permission = names
        self.colorDict = color_dict
        # create instance of SORT
        self.mot_tracker = Sort(max_age=10, min_hits=2)
        self.countArea = None
        self.running_flag = 0
        self.pause_flag = 0
        self.videoList = []
        self.last_max_id = 0
        self.history = {} 
        self.sin_runningFlag.connect(self.update_flag)
        self.sin_videoList.connect(self.update_videoList)
        self.sin_countArea.connect(self.update_countArea)
        self.sin_pauseFlag.connect(self.update_pauseFlag)
        self.save_dir = "results"
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
    def run(self):
        for video in self.videoList:
            self.last_max_id = 0
            cap = cv2.VideoCapture(video)
            out =  cv2.VideoWriter(os.path.join(self.save_dir,video.split("/")[-1]), cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (1920, 1080))
            frame_count = 0
            while cap.isOpened():
                #To print(frame_count) which is Frame Per Second:-
                if self.running_flag:
                    if not self.pause_flag:
                        ret, frame = cap.read()
                        if ret:
                            if frame_count % 3 == 0:
                                a1 = time.time()
                                frame = self.counter(self.permission, self.colorDict, frame,np.array(self.countArea), self.mot_tracker, video)
                                self.sin_counterResult.emit(frame)
                                out.write(frame)
                                a2 = time.time()
                                print(f"fps: {1 / (a2 - a1):.2f}")
                            frame_count += 1
                        else:
                            break
                    else:
                        time.sleep(0.1)
                else:
                    break
            #For Restart Count for Each Video:-
            KalmanBoxTracker.count = 0
            cap.release()
            out.release()
            if not self.running_flag:
                break
        if self.running_flag:
            self.sin_done.emit(1)
    def update_pauseFlag(self,flag):
        self.pause_flag = flag
    def update_flag(self,flag):
        self.running_flag = flag
    def update_videoList(self, videoList):
        print("Update videoList!")
        self.videoList = videoList
    def update_countArea(self,Area):
        print("Update countArea!")
        self.countArea = Area
    def counter(self, permission, colorDict, frame, CountArea, mot_tracker, videoName):
        #To Painting the Area:-
        AreaBound = [min(CountArea[:, 0]), min(CountArea[:, 1]), max(CountArea[:, 0]), max(CountArea[:, 1])]
        painting = np.zeros((AreaBound[3] - AreaBound[1], AreaBound[2] - AreaBound[0]), dtype=np.uint8)
        CountArea_mini = CountArea - AreaBound[0:2]
        cv2.fillConvexPoly(painting, CountArea_mini, (1,))
        objects = yolo_prediction(self.model,self.device,frame,self.class_names)
        objects = filter(lambda x: x[0] in permission, objects)
        objects = filter(lambda x: x[1] > 0.5,objects)
        objects = list(filter(lambda x: pointInCountArea(painting, AreaBound, [int(x[2][0]), int(x[2][1] + x[2][3] / 2)]),objects))
        #To Filter out repeat bbox:-
        objects = filiter_out_repeat(objects)
        detections = []
        for item in objects:
            detections.append([int(item[2][0] - item[2][2] / 2),
                               int(item[2][1] - item[2][3] / 2),
                               int(item[2][0] + item[2][2] / 2),
                               int(item[2][1] + item[2][3] / 2),
                               item[1]])
        track_bbs_ids = mot_tracker.update(np.array(detections))
#==============================================================================================================
#For Painting an Area:-
#==============================================================================================================
        for i in range(len(CountArea)):
            cv2.line(frame, tuple(CountArea[i]), tuple(CountArea[(i + 1) % (len(CountArea))]), (0, 0, 255), 2)
        if len(track_bbs_ids) > 0:
            for bb in track_bbs_ids:    #add all bbox to history
                id = int(bb[-1])
                objectName = get_objName(bb, objects)
                if id not in self.history.keys():  #add new id
                    self.history[id] = {}
                    self.history[id]["no_update_count"] = 0
                    self.history[id]["his"] = []
                    self.history[id]["his"].append(objectName)
                else:
                    self.history[id]["no_update_count"] = 0
                    self.history[id]["his"].append(objectName)
        for i, item in enumerate(track_bbs_ids):
            bb = list(map(lambda x: int(x), item))
            id = bb[-1]
            x1, y1, x2, y2 = bb[:4]
            his = self.history[id]["his"]
            result = {}
            for i in set(his):
                result[i] = his.count(i)
            res = sorted(result.items(), key=lambda d: d[1], reverse=True)
            objectName = res[0][0]
            boxColor = colorDict[objectName]
            cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, thickness=2)
            cv2.putText(frame, str(id) + "_" + objectName, (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        boxColor,
                        thickness=2)
        counter_results = []
        videoName = videoName.split('/')[-1]
        removed_id_list = []
        for id in self.history.keys():    #extract id after tracking
            self.history[id]["no_update_count"] += 1
            if  self.history[id]["no_update_count"] > 5:
                his = self.history[id]["his"]
                result = {}
                for i in set(his):
                    result[i] = his.count(i)
                res = sorted(result.items(), key=lambda d: d[1], reverse=True)
                objectName = res[0][0]
                counter_results.append([videoName,id,objectName])
                #del id
                removed_id_list.append(id)
        for id in removed_id_list:
            _ = self.history.pop(id)
        if len(counter_results):
            self.sin_counter_results.emit(counter_results)
        return frame
    def emit_timeCode(self,time_code):
        self.sin_timeCode.emit(time_code)
def getTwoDimensionListIndex(L,value,pos):
    for i in range(len(L)):
        if L[i][pos] == value:
            return i
    return -1
def filiter_out_repeat(objects):
    objects = sorted(objects,key=lambda x: x[1])
    l = len(objects)
    new_objects = []
    if l > 1:
        for i in range(l-1):
            flag = 0
            for j in range(i+1,l):
                x_i, y_i, w_i, h_i = objects[i][2]
                x_j, y_j, w_j, h_j = objects[j][2]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2), int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2), int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1,box2) >= 0.7:
                    flag = 1
                    break
            #if No repeatation then:-
            if not flag:
                new_objects.append(objects[i])
        #For Adding the last one object:-
        new_objects.append(objects[-1])
    else:
        return objects
    return list(tuple(new_objects))
def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou
def get_objName(item,objects):
    iou_list = []
    for i,object in enumerate(objects):
        x, y, w, h = object[2]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0]
def pointInCountArea(painting, AreaBound, point):
    h,w = painting.shape[:2]
    point = np.array(point)
    point = point - AreaBound[:2]
    if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
        return 0
    else:
        return painting[point[1],point[0]]
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
#================================================================================================
#Here the Below Code lines are For Reading a Video framewise (Taking it as image through opencv)
#================================================================================================
def yolo_prediction(model, device, image,class_names):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imgs = transforms.ToTensor()(Image.fromarray(image))
    c, h, w = imgs.shape
    img_sacle = [w / 416, h / 416, w / 416, h / 416]
    imgs = resize(imgs, 416)
    imgs = imgs.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.45)
    # print(outputs)
    objects = []
    try:
        outputs = outputs[0].cpu().data
        for i, output in enumerate(outputs):
            item = []
            item.append(class_names[int(output[-1])])
            item.append(float(output[4]))
            box = [int(value * img_sacle[i]) for i, value in enumerate(output[:4])]
            x1,y1,x2,y2 = box
            x = int((x2+x1)/2)
            y = int((y1+y2)/2)
            w = x2-x1
            h = y2-y1
            item.append([x,y,w,h])
            objects.append(item)
    except:
        pass
    return objects
#=============================================================================================
#Constructs module list of layer blocks from module configuration in module_defs:-
#=============================================================================================
def create_modules(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            #To Define detection layer:-
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        #For Register module list and number of output filters:-
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list
#===========================================================================================
#nn.Upsample is deprecated:-
#===========================================================================================
class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
#===========================================================================================
#Placeholder for 'route' and 'shortcut' layers:-
#===========================================================================================
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
#===================================================================================
#Detection Layer Code:-
#===================================================================================
class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
    def forward(self, x, targets=None, img_dim=None):
        #For Tensors for cuda support:-
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        #To Get Outputs:-
        x = torch.sigmoid(prediction[..., 0])  #Center x
        y = torch.sigmoid(prediction[..., 1])  #Center y
        w = prediction[..., 2]  #Width
        h = prediction[..., 3]  #Height
        pred_conf = torch.sigmoid(prediction[..., 4])  #Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  #Cls pred.
        #If grid size does not match current we compute new offsets:-
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        #Add offset and scale with anchors:-
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            #Loss:-Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            #Some Metrics Stuff:-
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            return output, total_loss
#===================================================================
#Yolov3 Object Detection Model Which Can detect upto 80 different Model:-
#===================================================================
class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)
#=========================================================================================================
#Here is the Use of Weights File by which we can train the images of objects and passes to the next one:-
#=========================================================================================================
    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file:-
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  #First five are header values:-
            self.header_info = header  #Needed to write header when saving weights:-
            self.seen = header[3]  #Number of images seen during training:-
            weights = np.fromfile(f, dtype=np.float32)  #The rest are weights:-
        #For Establish cutoff for loading backbone Weights:-
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance:-
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias:-
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight:-
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean:-
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var:-
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias:-
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights:-
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
    def save_darknet_weights(self, path, cutoff=-1):
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        # Iterate through layers:-
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first:-
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias:-
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights:-
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()
#===========================================================================
#Now Whole Code is Written For How GUI is Generated and How it Works:-
#===========================================================================
class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1203, 554)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_count = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_count.setGeometry(QtCore.QRect(990, 10, 211, 341))
        self.groupBox_count.setObjectName("groupBox_count")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_count)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_count = QtWidgets.QGridLayout()
        self.gridLayout_count.setContentsMargins(2, 2, 2, 2)
        self.gridLayout_count.setSpacing(6)
        self.gridLayout_count.setObjectName("gridLayout_count")
        self.label_truck = QtWidgets.QLabel(self.groupBox_count)
        self.label_truck.setObjectName("label_truck")
        self.gridLayout_count.addWidget(self.label_truck, 2, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_7 = QtWidgets.QLabel(self.groupBox_count)
        self.label_7.setObjectName("label_7")
        self.gridLayout_count.addWidget(self.label_7, 4, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_5 = QtWidgets.QLabel(self.groupBox_count)
        self.label_5.setObjectName("label_5")
        self.gridLayout_count.addWidget(self.label_5, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_6 = QtWidgets.QLabel(self.groupBox_count)
        self.label_6.setObjectName("label_6")
        self.gridLayout_count.addWidget(self.label_6, 3, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_motorbike = QtWidgets.QLabel(self.groupBox_count)
        self.label_motorbike.setObjectName("label_motorbike")
        self.gridLayout_count.addWidget(self.label_motorbike, 3, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_bus = QtWidgets.QLabel(self.groupBox_count)
        self.label_bus.setObjectName("label_bus")
        self.gridLayout_count.addWidget(self.label_bus, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_bicycle = QtWidgets.QLabel(self.groupBox_count)
        self.label_bicycle.setObjectName("label_bicycle")
        self.gridLayout_count.addWidget(self.label_bicycle, 4, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_12 = QtWidgets.QLabel(self.groupBox_count)
        self.label_12.setObjectName("label_12")
        self.gridLayout_count.addWidget(self.label_12, 5, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_3 = QtWidgets.QLabel(self.groupBox_count)
        self.label_3.setObjectName("label_3")
        self.gridLayout_count.addWidget(self.label_3, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_sum = QtWidgets.QLabel(self.groupBox_count)
        self.label_sum.setObjectName("label_sum")
        self.gridLayout_count.addWidget(self.label_sum, 5, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_car = QtWidgets.QLabel(self.groupBox_count)
        self.label_car.setObjectName("label_car")
        self.gridLayout_count.addWidget(self.label_car, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_4 = QtWidgets.QLabel(self.groupBox_count)
        self.label_4.setObjectName("label_4")
        self.gridLayout_count.addWidget(self.label_4, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.verticalLayout_2.addLayout(self.gridLayout_count)
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 960, 540))
        self.label_image.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.label_image.setText("")
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setObjectName("label_image")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(1020, 360, 151, 181))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_openVideo = QtWidgets.QPushButton(self.widget)
        self.pushButton_openVideo.setObjectName("pushButton_openVideo")
        self.verticalLayout.addWidget(self.pushButton_openVideo)
        self.pushButton_selectArea = QtWidgets.QPushButton(self.widget)
        self.pushButton_selectArea.setObjectName("pushButton_selectArea")
        self.verticalLayout.addWidget(self.pushButton_selectArea)
        self.pushButton_start = QtWidgets.QPushButton(self.widget)
        self.pushButton_start.setObjectName("pushButton_start")
        self.verticalLayout.addWidget(self.pushButton_start)
        self.pushButton_pause = QtWidgets.QPushButton(self.widget)
        self.pushButton_pause.setObjectName("pushButton_pause")
        self.verticalLayout.addWidget(self.pushButton_pause)
        mainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Car Counter"))
        self.groupBox_count.setTitle(_translate("mainWindow", "Counting Results"))
        self.label_truck.setText(_translate("mainWindow", "0"))
        self.label_7.setText(_translate("mainWindow", "Bicycle"))
        self.label_5.setText(_translate("mainWindow", "Truck"))
        self.label_6.setText(_translate("mainWindow", "Motorbike"))
        self.label_motorbike.setText(_translate("mainWindow", "0"))
        self.label_bus.setText(_translate("mainWindow", "0"))
        self.label_bicycle.setText(_translate("mainWindow", "0"))
        self.label_12.setText(_translate("mainWindow", "Sum"))
        self.label_3.setText(_translate("mainWindow", "Car"))
        self.label_sum.setText(_translate("mainWindow", "0"))
        self.label_car.setText(_translate("mainWindow", "0"))
        self.label_4.setText(_translate("mainWindow", "Bus"))
        self.pushButton_openVideo.setText(_translate("mainWindow", "Open Video"))
        self.pushButton_selectArea.setText(_translate("mainWindow", "Please Select the Area"))
        self.pushButton_start.setText(_translate("mainWindow", "Please Start"))
        self.pushButton_pause.setText(_translate("mainWindow", "You Can Pause it"))
#===================================================================================
#Main Class Loading While We write python Main.py on Anaconda Prompt:-
#===================================================================================
class App(QMainWindow,Ui_mainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.setupUi(self)
        self.label_image_size = (self.label_image.geometry().width(),self.label_image.geometry().height())
        self.video = None
        self.exampleImage = None
        self.imgScale = None
        self.get_points_flag = 0
        self.countArea = []
        self.road_code = None
        self.time_code = None
        self.show_label = names
        #Button function:-
        self.pushButton_selectArea.clicked.connect(self.select_area)
        self.pushButton_openVideo.clicked.connect(self.open_video)
        self.pushButton_start.clicked.connect(self.start_count)
        self.pushButton_pause.clicked.connect(self.pause)
        self.label_image.mouseDoubleClickEvent = self.get_points
        self.pushButton_selectArea.setEnabled(False)
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setEnabled(False)
        #Some flags:-
        self.running_flag = 0
        self.pause_flag = 0
        self.counter_thread_start_flag = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_config = "config/coco.data"
        weights_path = "weights/yolov3.weights"
        model_def = "config/yolov3.cfg"
        data_config = parse_data_config(data_config)
        self.yolo_class_names = load_classes(data_config["names"])
        #For Initiate model:-
        print("Loading model ...")
        self.yolo_model = Darknet(model_def).to(self.device)
        if weights_path.endswith(".weights"):
            #For Loading the darknet Weights:-
            self.yolo_model.load_darknet_weights(weights_path)
        else:
            #For Loading the Checkpoint Weights:-
            self.yolo_model.load_state_dict(torch.load(weights_path))
        #For Counter Thread:-
        self.counterThread = CounterThread(self.yolo_model,self.yolo_class_names,self.device)
        self.counterThread.sin_counterResult.connect(self.show_image_label)
        self.counterThread.sin_done.connect(self.done)
        self.counterThread.sin_counter_results.connect(self.update_counter_results)
    def open_video(self):
        openfile_name = QFileDialog.getOpenFileName(self,'Open video','','Video files(*.avi , *.mp4)')
        self.videoList = [openfile_name[0]]
        vid = cv2.VideoCapture(self.videoList[0])
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
                self.imgScale = np.array(frame.shape[:2]) / [self.label_image_size[1], self.label_image_size[0]]
                vid.release()
                break
        self.pushButton_selectArea.setEnabled(True)
        self.pushButton_start.setText("Please Start the Video")
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setText("You Can Pause it")
        self.pushButton_pause.setEnabled(False)
        #clear counting results
        KalmanBoxTracker.count = 0
        self.label_sum.setText("0")
        self.label_sum.repaint()
    def get_points(self, event):
        if self.get_points_flag:
            x = event.x()
            y = event.y()
            self.countArea.append([int(x*self.imgScale[1]),int(y*self.imgScale[0])])
            exampleImageWithArea = copy.deepcopy(self.exampleImage)
            for point in self.countArea:
                exampleImageWithArea[point[1]-10:point[1]+10,point[0]-10:point[0]+10] = (0,255,255)
            cv2.fillConvexPoly(exampleImageWithArea, np.array(self.countArea), (0,0,255))
            self.show_image_label(exampleImageWithArea)
        print(self.countArea)
    def select_area(self):
        #To Change the Area needs update exampleImage
        if self.counter_thread_start_flag:
            ret, frame = self.videoCapture.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
        if not self.get_points_flag:
            self.pushButton_selectArea.setText("Please Submit the Area")
            self.get_points_flag = 1
            self.countArea = []
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_start.setEnabled(False)
        else:
            self.pushButton_selectArea.setText("Please Select the Area")
            self.get_points_flag = 0
            exampleImage = copy.deepcopy(self.exampleImage)
            #To Painting an Area:-
            for i in range(len(self.countArea)):
                cv2.line(exampleImage, tuple(self.countArea[i]), tuple(self.countArea[(i + 1) % (len(self.countArea))]), (0, 0, 255), 2)
            self.show_image_label(exampleImage)
            #To Enable start button
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(True)
    def show_image_label(self, img_np):
        img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.label_image_size)
        frame = QImage(img_np, self.label_image_size[0], self.label_image_size[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.label_image.setPixmap(pix)
        self.label_image.repaint()
    def start_count(self):
        if self.running_flag == 0:
            #To Clear count and display:-
            KalmanBoxTracker.count = 0
            for item in self.show_label:
                vars(self)[f"label_{item}"].setText('0')
            #To Clear Final file:-
            with open("Final/Final.txt", "w") as f:
                pass
            #To Start:-
            self.running_flag = 1
            self.pause_flag = 0
            self.pushButton_start.setText("You Can Stop it")
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_selectArea.setEnabled(False)
            #To Emit new parameter to counter thread:-
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.counterThread.sin_countArea.emit(self.countArea)
            self.counterThread.sin_videoList.emit(self.videoList)
            #To Start the counter thread:-
            self.counterThread.start()
            self.pushButton_pause.setEnabled(True)
        elif self.running_flag == 1:  #push pause button
            #To Stop the System:-
            self.running_flag = 0
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_selectArea.setEnabled(True)
            self.pushButton_start.setText("Please Start")
    def done(self,sin):
        if sin == 1:
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(False)
            self.pushButton_start.setText("Start")
    def update_counter_results(self,counter_results):
        with open("Final/Final.txt", "a") as f:
            for i, result in enumerate(counter_results):
                label_var = vars(self)[f"label_{result[2]}"]
                label_var.setText(str(int(label_var.text())+1))
                label_var.repaint()
                label_sum_var = vars(self)[f"label_sum"]
                label_sum_var.setText(str(int(label_sum_var.text()) + 1))
                label_sum_var.repaint()
                f.writelines(' '.join(map(lambda x: str(x),result)))
                f.write(("\n"))
        # print("************************************************",len(counter_results))
    def pause(self):
        if self.pause_flag == 0:
            self.pause_flag = 1
            self.pushButton_pause.setText("Continue")
            self.pushButton_start.setEnabled(False)
        else:
            self.pause_flag = 0
            self.pushButton_pause.setText("Pause")
            self.pushButton_start.setEnabled(True)
        self.counterThread.sin_pauseFlag.emit(self.pause_flag)
#================================================================        
#Main Function of Project Running:-
#================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = App()
    myWin.show()
    sys.exit(app.exec_())
#================================================================