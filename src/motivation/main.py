"""
In this file, we implement all the helper functions that are needed to prove the motivation section.
"""
import os
import shutil
from data.data_loader import Loader
from udfs.yolov5_wrapper import YOLOv5Wrapper
from src.motivation.yolo_wrapper import YoloWrapper
from src.motivation.tasti_wrapper import MotivationConfig, MotivationTasti
from src.motivation.eko_wrapper import EKOConfig, EKO
from src.motivation.resnet_wrapper import ResnetWrapper
from src.motivation.maskrcnn_wrapper import MaskRCNNWrapper

### import the queries
from benchmarks.stanford.tasti.tasti.seiden.queries.queries import NightStreetAggregateQuery, \
    NightStreetAveragePositionAggregateQuery, \
    NightStreetSUPGPrecisionQuery, \
    NightStreetSUPGRecallQuery

from src.system_architecture.alternate import EKO_alternate
from src.system_architecture.parameter_search import EKOPSConfig, EKO_PS

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from seiden_utils.file_utils import load_json

import torch
import torchvision
from tqdm import tqdm
import time
from sklearn.svm import SVC
import numpy as np


class SystemTool:
    def __init__(self):
        self.annotation = ''
        # self.directory = 'D:/Projects/PyhtonProjects/thesis/tasti_data/cache'
        self.directory = '/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache'

    def delete_files_in_directory(self):
        """
            删除指定目录下的所有文件和子目录。

            Args:
                directory (str): 要删除文件的目录路径。

            Returns:
                None
            """
        directory = self.directory
        try:
            # 确保目录存在并且是一个目录
            if not os.path.exists(directory) or not os.path.isdir(directory):
                print(f"Error: 目录 '{directory}' 不存在或不是一个有效目录。")
                return

            # 获取目录下的所有文件和子目录
            items = os.listdir(directory)

            # 遍历所有文件和子目录
            for item in items:
                item_path = os.path.join(directory, item)

                # 如果是文件，则直接删除
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"删除文件: {item_path}")

                # 如果是目录，则递归删除其内部的所有文件和子目录
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"删除目录及其内容: {item_path}")

            print(f"所有文件和目录已成功删除: {directory}")

        except Exception as e:
            print(f"Error: 删除文件时发生异常: {str(e)}")

    def save_tasti_times(self, model_name, tasti_times, num):
        # 构建保存的字符串格式
        data_str = f"'{model_name}' : {tasti_times}\n"
        # 文件路径
        # file_path = f'/home/wangshuo_20/pythonpr/seiden_ws/data/RQ{num}_data/RQ{num}.txt'
        file_path = f'/home/wangshuo_20/pythonpr/seiden_ws/data/RQ{num}_data/RQ{num}.txt'
        # 确保文件目录存在，若不存在则创建
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 打开文件，追加写入数据
        with open(file_path, 'a') as file:
            file.write(data_str)

    def save_precison_recall(self, query_pattern, result, num):
        # 构建保存的字符串格式
        data_str = f"'{query_pattern}' : {result['precision']}  ,{result['recall']}  ,{result['actual_estimate']}\n"
        # 文件路径
        file_path = f'/home/wangshuo_20/pythonpr/seiden_ws/data/RQ{num}_data/RQ{num}.txt'
        # 确保文件目录存在，若不存在则创建
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 打开文件，追加写入数据
        with open(file_path, 'a') as file:
            file.write(data_str)

    def save_agg(self, query_pattern, result, num):
        # 构建保存的字符串格式
        data_str = f"'{query_pattern}' : {result['initial_estimate']} ,{result['blazeit_estimate']} ,{result['actual_estimate']}  \n"
        # 文件路径
        file_path = f'/home/wangshuo_20/pythonpr/seiden_ws/data/RQ{num}_data/RQ{num}.txt'
        # 确保文件目录存在，若不存在则创建
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 打开文件，追加写入数据
        with open(file_path, 'a') as file:
            file.write(data_str)


##################################
#### Evaluation Code #############
##################################

def evaluate_object_detection(gt_file, dt_file):
    cocoGT = COCO(gt_file)
    ## open the dt_file and put in annotations
    dt = load_json(dt_file)
    cocoDT = cocoGT.loadRes(dt['annotations'])
    cocoEVAL = COCOeval(cocoGT, cocoDT, 'bbox')
    cocoEVAL.evaluate()
    cocoEVAL.accumulate()
    cocoEVAL.summarize()


###################################
### Code for Query Execution ######
###################################
THROUGHPUT = 1 / 140


def query_process_aggregate_yolov(result):
    st = time.perf_counter()
    times = []
    # print(result)
    ct = 0
    index_cahce = result['annotations']
    for its in index_cahce.values():
        ct += len(its)
    et = time.perf_counter()
    times.append(et - st)
    print('[cars_number]: ', ct)
    return ct


def query_process_aggregate(index, images=None):
    st = time.perf_counter()
    times = []
    print('[motivation_main]')
    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.1, confidence=0.05, images=images)
    times.append(result['nb_samples'])

    et = time.perf_counter()
    times.append(et - st)

    return times, result


def query_process_precision(index, dnn_invocation=1000, images=None):
    st = time.perf_counter()
    times = []

    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(dnn_invocation, None, images)
    # times.append((result['precision'], result['recall']))

    et = time.perf_counter()
    times.append(et - st)
    print(times)
    return times, result


def query_process_recall(index):
    st = time.perf_counter()
    times = []

    query = NightStreetSUPGRecallQuery(index)
    result = query.execute_metrics(1000)
    times.append((result['precision'], result['recall']))

    et = time.perf_counter()
    times.append(et - st + 1000 * THROUGHPUT)

    return times, result


def query_process1(index):
    times = []

    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.01, confidence=0.05)
    times.append(result['nb_samples'])

    result = query.execute_metrics(err_tol=0.1, confidence=0.05)
    times.append(result['nb_samples'])

    result = query.execute_metrics(err_tol=1, confidence=0.05)
    times.append(result['nb_samples'])

    return times


def query_process(index):
    times = []

    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.01, confidence=0.05)
    times.append(result['nb_samples'])

    im_size = 360
    query = NightStreetAveragePositionAggregateQuery(index, im_size)
    result = query.execute_metrics(err_tol=0.001, confidence=0.05)
    times.append(result['nb_samples'])

    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(7000)
    times.append((result['precision'], result['recall']))

    return times


def query_process2(index):
    times = []
    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(5000)
    times.append((result['precision'], result['recall']))

    return times


###################################
### Code for index construction ###
###################################

def load_dataset(video_name):
    ### load video to memory
    loader = Loader()
    video_fp = os.path.join('/home/wangshuo_20/pythonpr/thesis_data/video_data/', video_name)
    # video_fp = os.path.join('D:/Projects/PyhtonProjects/thesis/video_data/', video_name)
    images = loader.load_video(video_fp)
    return images


def load_dataset_videos(video_name):
    ### load video to memory
    loader = Loader()
    video_fp = os.path.join('/home/wangshuo_20/pythonpr/thesis_data/video_data/', video_name)
    # video_fp = os.path.join('D:/Projects/PyhtonProjects/thesis/video_data/', video_name)
    images = loader.load_videos(video_fp, video_name)
    return images


def execute_svm(images, image_size=None):
    ## how will we transform this matrix to fit the image size?
    width, height = images.shape[1], images.shape[2]
    if image_size is not None:
        width_division = width // image_size
        height_division = height // image_size
        new_images = images[:, ::width_division, ::height_division, :]
    else:
        new_images = images
    new_images = new_images.reshape(len(new_images), -1)

    train_images = new_images[::1000]
    y_random = np.random.randint(2, size=len(train_images))

    ### we need to try out the svm model
    clf = SVC(gamma='auto')
    clf.fit(train_images, y_random)

    st = time.perf_counter()
    output = clf.predict(new_images)
    et = time.perf_counter()

    return et - st


def execute_resnet(images, image_size=None):
    resnet = ResnetWrapper()
    output = resnet.inference(images, image_size)

    return output


def execute_yolo(images):
    yolo = YOLOv5Wrapper()
    output = yolo.inference(images)

    return output


def execute_yolo2(images):
    yolo = YoloWrapper()
    output = yolo.inference_o(images)
    return output


def execute_maskrcnn(images, batch_size=8):
    mask = MaskRCNNWrapper()

    output = mask.inference(images, batch_size=batch_size)

    return output


def execute_maskrcnn_features(images, batch_size=8):
    mask = MaskRCNNWrapper()

    output = mask.inference_features(images, batch_size=batch_size)
    return output


def execute_eko(images, video_name, nb_buckets=7000, dist_param=0.1, temp_param=0.9):
    ekoconfig = EKOConfig(video_name, nb_buckets=nb_buckets, dist_param=dist_param, temp_param=temp_param)
    eko = EKO(ekoconfig, images)
    eko.init()

    return eko


def execute_ekoalt(images, video_name, category='car', nb_buckets=7000):
    ekoconfig = EKOPSConfig(video_name, category=category, nb_buckets=nb_buckets)
    ekoalt = EKO_alternate(ekoconfig, images)
    ekoalt.init()

    return ekoalt


def execute_tastipt(images, video_name, category='car', redo=False, image_size=None, nb_buckets=7000):
    ### call tasti -- init, bucket... we must exclude dataloading time, we must include execution time (or at least count)
    print('[nb_buckets]', nb_buckets)
    do_train = False
    do_infer = redo
    motivationconfig = MotivationConfig(video_name, do_train, do_infer, image_size=image_size, nb_buckets=nb_buckets,
                                        category=category)
    motivationtasti = MotivationTasti(motivationconfig, images)
    motivationtasti.init()

    return motivationtasti


def execute_tasti(images, video_name, nb_buckets=7000, nb_train=1000):
    ### call tasti -- init, bucket... we must exclude dataloading time, we must include execution time (or at least count)
    do_train = True
    print('[exe of nb_buckets]', nb_buckets)
    motivationconfig = MotivationConfig(video_name, do_train, do_infer=True, nb_buckets=nb_buckets, nb_train=nb_train)
    motivationtasti = MotivationTasti(motivationconfig, images)
    motivationtasti.init()

    return motivationtasti
