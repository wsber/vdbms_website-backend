"""
This file consists of the decompressionModule class that works with compressed videos to make them decompressed.
Some further optimizations could be possible (such as utilization of multiple threads, but at the moment everything is serial)

@Jaeho Bang
"""

import cv2
import numpy as np
import os
import time
from seiden_utils.logger import Logger
from PIL import Image
from tqdm import tqdm
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


# 以上代码定义了一个名为 DecompressionModule 的类，用于处理视频文件的解压缩和转换成图像序列。
class DecompressionModule:
    def __init__(self):
        self.image_matrix = None
        self.video_stats = {}  # This will keep data of all the videos that have been parsed.. will not keep the image matrix only meta-data
        self.logger = Logger()
        self.curr_video = ''

    def reset(self):
        self.image_matrix = None

    # add_meta_data方法用于向video_stats中添加视频的元数据，包括帧数、宽度、高度和帧率等。
    def add_meta_data(self, path, frame_count, width, height, fps):
        if path in self.video_stats:
            self.video_stats[path]['frame_count'] += frame_count
        else:
            self.video_stats[path] = {}
            self.video_stats[path]['fps'] = fps
            self.video_stats[path]['width'] = width
            self.video_stats[path]['height'] = height
            self.video_stats[path]['frame_count'] = frame_count

    # get_frame_count 方法用于获取当前视频的帧数。
    def get_frame_count(self):
        return self.video_stats[self.curr_video]['frame_count']

    def get_iframes(self, path, frame_count_limit=60000):
        pass

    # convert_and_save方法用于将视频转换为图像序列并保存到指定目录。
    def convert_and_save(self, load_directory, save_directory):
        vid_ = cv2.VideoCapture(load_directory)
        frame_count = int(vid_.get(cv2.CAP_PROP_FRAME_COUNT))
        ## let's cap the frame_count to 300k
        ### no when we do convert and save, we load one and save and repeat

        for i in tqdm(range(frame_count)):
            success, image = vid_.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            save_filename = os.path.join(save_directory, '{:09d}.jpg'.format(i))
            image.save(save_filename)

        print(f"Saved {load_directory} to {save_directory}")

    def convert_and_save_by_frame_Id(self, load_directory, save_directory, frame_ids, video_uuid_name):
        vid_ = cv2.VideoCapture(load_directory)
        frame_count = int(vid_.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid_.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new dimensions
        new_width = width // 2
        new_height = height // 2
        ## let's cap the frame_count to 300k
        ### no when we do convert and save, we load one and save and repeat
        k = 0
        update_step_length = int(float(frame_count) * 0.025)
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "algorithm_type_group",
            {
                "type": "algorithm.update",
                "algorithm_type": 'extract_frames',
                "consumer_type": 'algorithm_type'
            }
        )
        imagesName = []
        for i in tqdm(range(frame_count)):
            success, image = vid_.read()
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            if i % update_step_length == 0 or i == frame_count - 1:
                progress = (i + 1) / frame_count * 100
                # print('[progress]: ', progress)
                async_to_sync(channel_layer.group_send)(
                    "progress_group",
                    {
                        "type": "progress.update",
                        "progress": progress,
                        "algorithm_type": 'extract_frames'
                    }
                )
            if i == frame_ids[k]:
                save_filename = os.path.join(save_directory, '{:09d}.jpg'.format(k))
                imagesName.append(fr'media/outData/{video_uuid_name}/' + '{:09d}.jpg'.format(k))
                image.save(save_filename)
                k += 1
                if k == len(frame_ids):
                    async_to_sync(channel_layer.group_send)(
                        "progress_group",
                        {
                            "type": "progress.update",
                            "progress": 100,
                            "algorithm_type": 'extract_frames'
                        }
                    )
                    break

        print(f"Saved {load_directory} to {save_directory}")
        return imagesName

    # convert2images 方法用于将视频转换为图像矩阵，并返回图像矩阵。
    # 在转换过程中，会根据指定的帧数限制和尺寸对图像进行调整，并将其存储在image_matrix中。
    def convert2images(self, path, frame_count_limit=300000, size=[234, 416]):
        self.vid_ = cv2.VideoCapture(path)
        if (self.vid_.isOpened() == False):
            self.logger.error(f"Error opening video {path}")
            raise ValueError

        self.curr_video = path

        frame_count = min(self.vid_.get(cv2.CAP_PROP_FRAME_COUNT), frame_count_limit)
        width = self.vid_.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.vid_.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.vid_.get(cv2.CAP_PROP_FPS)
        channels = 3

        if size is not None:
            height = size[0]
            width = size[1]

        self.add_meta_data(path, frame_count, width, height, fps)

        assert (frame_count == int(frame_count))
        assert (width == int(width))
        assert (height == int(height))

        frame_count = int(frame_count)
        width = int(width)
        height = int(height)

        self.logger.info(f"meta data of the video {path} is {frame_count, height, width, channels}")
        self.image_matrix = np.ndarray(shape=(frame_count, height, width, channels), dtype=np.uint8)

        error_indices = []
        channel_layer = get_channel_layer()
        update_step_length = int(float(frame_count) * 0.025)
        async_to_sync(channel_layer.group_send)(
            "algorithm_type_group",
            {
                "type": "algorithm.update",
                "algorithm_type": 'data_load',
                "consumer_type": 'algorithm_type'
            }
        )
        print('[update_step_length]: ', update_step_length)
        for i in tqdm(range(frame_count)):
            success, image = self.vid_.read()
            if not success:
                print(f"Image {i} retrieval has failed")
                error_indices.append(i)
            else:
                ### need to resize the image matrix
                image = cv2.resize(image, (width, height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.image_matrix[i, :, :, :] = image  # stored in rgb format
            if i % update_step_length == 0 or i == frame_count - 1:
                progress = (i + 1) / frame_count * 100
                # print('[progress]: ', progress)
                async_to_sync(channel_layer.group_send)(
                    "progress_group",
                    {
                        "type": "progress.update",
                        "progress": progress,
                        "algorithm_type": 'data_load'
                    }
                )

        ### let's do error fixing
        """
        if len(error_indices) != 0:
            error_indices = sorted(error_indices) ## make the lower one come first
            for i, error_index in enumerate(error_indices):
                ## the reason we delete the error_index - i is because this deleting mechanism is destructive
                ## since we have already sorted the error_indices array, we are guaranteed that we have deleted the number of elements before hand
                ## therefore, the total length of the image matrix has been decreased
                self.image_matrix = np.delete(self.image_matrix, error_index - i, axis = 0)
        
        ## => I am not sure how to do this in a memory efficient way
        """

        return self.image_matrix

    def capture_videos_in_folder(self, folder_path):
        video_capture_objects = {}  # 用来保存VideoCapture对象的字典或列表

        # 获取指定文件夹下所有视频文件的路径
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        # 创建VideoCapture对象并保存到字典中
        for video_file in tqdm(video_files, desc='Capturing Videos'):
            video_path = os.path.join(folder_path, video_file)
            capture = cv2.VideoCapture(video_path)

            if capture.isOpened():
                video_capture_objects[video_file] = capture
            else:
                print(f"Failed to open video file: {video_file}")
        return video_capture_objects

    def get_video_capture_objects(self, folder_path, video_name, frame_count_limit=300000, size=[234, 416]):
        # 指定视频文件所在的文件夹路径
        # folder_path = fr'D:/Projects/PyhtonProjects/thesis/video_data/{video_name}'
        folder_path = fr'/home/wangshuo_20/pythonpr/VDBMS_ws/media/{video_name}'
        # folder_path = fr'/home/wangshuo_20/pythonpr/thesis_data/video_data/{video_name}'
        path = os.path.join(folder_path, 'video')
        # 获取文件夹中所有视频的VideoCapture对象
        video_captures = self.capture_videos_in_folder(folder_path)
        all_image_matrices = []
        # 现在你可以使用video_captures字典中的VideoCapture对象来处理视频了
        for video_name, video_capture in tqdm(video_captures.items(), desc='convert Videos to image_matrix'):
            # 在这里你可以对每个视频进行处理，例如读取帧、显示视频等
            # print(f"Processing video: {video_name}")
            self.vid_ = video_capture
            frame_count = min(self.vid_.get(cv2.CAP_PROP_FRAME_COUNT), frame_count_limit)
            width = self.vid_.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.vid_.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.vid_.get(cv2.CAP_PROP_FPS)
            channels = 3
            if size is not None:
                height = size[0]
                width = size[1]
            self.add_meta_data(path, frame_count, width, height, fps)
            assert (frame_count == int(frame_count))
            assert (width == int(width))
            assert (height == int(height))
            frame_count = int(frame_count)
            width = int(width)
            height = int(height)

            self.image_matrix = np.ndarray(shape=(frame_count, height, width, channels), dtype=np.uint8)

            error_indices = []

            for i in tqdm(range(frame_count), desc='video processing'):
                success, image = self.vid_.read()
                if not success:
                    print(f"Image {i} retrieval has failed")
                    error_indices.append(i)
                else:
                    ### need to resize the image matrix
                    image = cv2.resize(image, (width, height))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.image_matrix[i, :, :, :] = image  # stored in rgb format
                    # print(self.image_matrix)
            all_image_matrices.extend(self.image_matrix)
        all_image_matrices = np.array(all_image_matrices)
        # print(all_image_matrices)
        return all_image_matrices


if __name__ == "__main__":
    eva_dir = os.path.abspath('../')
    data_dir = os.path.join(eva_dir, 'data', 'videos')
    dc = DecompressionModule()
    files = os.listdir(data_dir)

    full_name = os.path.join(data_dir, files[0])
    tic = time.time()
    print("--------------------------")
    print("Starting ", files[0])
    dc.convert2images(full_name)
    print("Finished conversion, total time taken is:", time.time() - tic, "seconds")
    print("Image matrix shape:", dc.image_matrix.shape)
