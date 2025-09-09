#-------------------------------------#
#       模型评估专用脚本 - event_map
#-------------------------------------#
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm  
import glob
from PIL import Image
from nets.yolo import YoloBody
from utils.utils import get_classes, preprocess_input, resize_image, cvtColor
from utils.utils_bbox import decode_outputs, non_max_suppression
from utils.event_json_dataloader import EventJsonDataset
from utils.utils_map import get_map

# 设置中文显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 定义事件数据集的类别名称
CLASS_NAMES = [
    "people", "car", "bicycle", "electric bicycle", 
    "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
]

class YOLO_MAP:
    def __init__(self, classes_path, phi, model_path, input_shape=[640, 640], confidence=0.5, nms_iou=0.6, 
                 letterbox_image=True, Cuda=True, map_mode=False, map_confidence=0.5, map_iou=0.5):
        #-----------------------------------------------#
        #   获得种类和先验框的数量
        #-----------------------------------------------#
        self.class_names = CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.input_shape = input_shape
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.map_mode = map_mode
        self.map_confidence = map_confidence
        self.map_iou = map_iou
        self.cuda = Cuda  # 修正变量名，保持一致

        #---------------------------------------------------#
        #   创建yolo模型
        #---------------------------------------------------#
        self.generate(YoloBody(self.num_classes, phi), model_path, Cuda)
        
        #---------------------------------------------------#
        #   预先创建存储结果的文件夹
        #---------------------------------------------------#
        self.result_path = 'map_out'
        self.gt_path = os.path.join(self.result_path, 'ground-truth')
        self.dr_path = os.path.join(self.result_path, 'detection-results')
        for path in [self.result_path, self.gt_path, self.dr_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def generate(self, model, model_path, Cuda):
        self.net = model
        #---------------------------------------------------#
        #   权值加载
        #---------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(model_path))
        
        if Cuda and torch.cuda.is_available():
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def detect_image(self, image, image_id):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错
        #---------------------------------------------------------#
        image = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            
            if results[0] is None:
                return

            # 修复结果索引问题
            top_label = np.array(results[0][:, 6], dtype = 'int32')  # 使用第6列作为类别索引
            top_conf = results[0][:, 4] * results[0][:, 5]  # 计算最终置信度
            top_boxes = results[0][:, :4]  # 获取边界框坐标

        #---------------------------------------------------------#
        #   生成检测结果文件
        #---------------------------------------------------------#
        det_results = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            left, top, right, bottom = box
            det_results.append(f"{predicted_class} {score} {int(left)} {int(top)} {int(right)} {int(bottom)}")
            
        # 保存检测结果到文件
        with open(os.path.join(self.dr_path, f"{image_id}.txt"), "w") as f:
            if det_results:
                f.write("\n".join(det_results))

    def get_gt_txt(self, image_id, boxes):
        #---------------------------------------------------------#
        #   生成真实框结果文件
        #---------------------------------------------------------#
        gt_results = []
        for box in boxes:
            # 确保box的格式正确 [x1, y1, x2, y2, class_id]
            if len(box) >= 5:
                x1, y1, x2, y2, class_id = box[:5]
                if 0 <= int(class_id) < len(self.class_names):
                    class_name = self.class_names[int(class_id)]
                    gt_results.append(f"{class_name} {int(x1)} {int(y1)} {int(x2)} {int(y2)}")
        
        # 保存真实框结果到文件
        with open(os.path.join(self.gt_path, f"{image_id}.txt"), "w") as f:
            if gt_results:
                f.write("\n".join(gt_results))

    def get_map_txt(self, json_path, image_root, batch_size=8):
        #---------------------------------------------------------#
        #   创建数据集加载器
        #---------------------------------------------------------#
        val_dataset = EventJsonDataset(
            json_path=json_path,
            image_root=image_root,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            epoch_length=1,
            mosaic=False,
            train=False
        )
        
        #---------------------------------------------------------#
        #   遍历数据集并生成结果文件
        #---------------------------------------------------------#
        print("开始生成mAP计算所需的文件...")
        # 直接遍历所有图像ID，不使用DataLoader返回的数据
        for image_id in tqdm(val_dataset.image_ids):
            try:
                # 获取图像和真实框
                image, gt_boxes = val_dataset.get_image_and_boxes(image_id)
                
                # 生成检测结果
                self.detect_image(image, image_id)
                
                # 生成真实框结果
                self.get_gt_txt(image_id, gt_boxes)
            except Exception as e:
                print(f"处理图像 {image_id} 时出错: {str(e)}")
        print("mAP计算所需的文件已生成完成")

if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/lhl/Git/yolox-pytorch-bilibili/logs/ep002-loss10.452-val_loss10.082.pth', help='模型权重文件路径')
    parser.add_argument('--json_path', type=str, default='/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/val.json', help='测试集JSON文件路径')
    parser.add_argument('--image_root', type=str, default='/home/lhl/Git/datasets/EvDET200K/Event_Frame/data', help='图像根目录')
    parser.add_argument('--phi', type=str, default='s', help='模型版本')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--map_iou', type=float, default=0.1, help='mAP计算的IoU阈值')
    args = parser.parse_args()

    # 初始化YOLO_MAP对象
    yolo_map = YOLO_MAP(
        classes_path='',  # 这里不使用，因为我们直接定义了CLASS_NAMES
        phi=args.phi,
        model_path=args.model_path,
        input_shape=[640, 640],
        confidence=0.1,
        nms_iou=0.6,
        letterbox_image=True,
        Cuda=True,
        map_mode=True,
        map_confidence=0.5,
        map_iou=args.map_iou
    )

    # 生成mAP计算所需的文件
    yolo_map.get_map_txt(args.json_path, args.image_root, args.batch_size)

    # 计算mAP
    print("开始计算mAP...")
    get_map(MINOVERLAP=args.map_iou, draw_plot=True)
    print("mAP计算完成！")