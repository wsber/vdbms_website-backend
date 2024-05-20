import torch
import torchvision
import tasti
import numpy as np
import os
import csv

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from tqdm.autonotebook import tqdm


class Index:
    def __init__(self, config):
        self.config = config
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='train'),
            self.target_dnn_callback
        )
        self.filepath = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden/{self.config.video_name}_cache.csv"
        # self.filepath = fr"D:/Projects/PyhtonProjects/thesis/tasti_data/cache/{self.config.video_name}/seiden/{self.config.video_name}_cache.csv"
        self.filepath_full = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden/{self.config.video_name}_cache_full.csv"
        # self.filepath_full = fr"D:/Projects/PyhtonProjects/thesis/tasti_data/cache/{self.config.video_name}/seiden/{self.config.video_name}_cache_full.csv"
        self.index_cache = self.read_csv_to_index(self.filepath)
        self.full_index_cache = self.read_csv_to_index(self.filepath_full)
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='train')
        self.seed = self.config.seed
        self.rand = np.random.RandomState(seed=self.seed)
        self.cache_dir = self.get_cache_dir()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块 GPU
        self.model.to(self.device)
        self.model.eval()
        print('[device]: ', self.device)
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                        'teddy bear', 'hair drier', 'toothbrush']

    def update_category(self, category):
        self.config.category = category
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='train')

    def get_cache_dir(self):
        raise NotImplementedError

    def get_num_workers(self):
        raise NotImplementedError

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        '''
        This allows you to override tasti.seiden_utils.DNNOutputCache if you already have the target dnn
        outputs available somewhere. Returning a list or another 1-D indexable element will work.
        '''
        return target_dnn_cache

    def is_close(self, a, b):
        '''
        Define your notion of "closeness" as described in the paper between records "a" and "b".
        Return a Boolean.
        '''
        raise NotImplementedError

    def get_target_dnn_dataset(self, train_or_test='train'):
        '''
        Define your target_dnn_dataset under the condition of "train_or_test".
        Return a torch.seiden_utils.data.Dataset object.
        '''
        raise NotImplementedError

    def get_embedding_dnn_dataset(self, train_or_test='train'):
        '''
        Define your embedding_dnn_dataset under the condition of "train_or_test".
        Return a torch.seiden_utils.data.Dataset object.
        '''
        raise NotImplementedError

    def get_target_dnn(self):
        '''
        Define your Target DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError

    def get_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError

    def get_pretrained_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        return self.get_pretrained_embedding_dnn()

    def target_dnn_callback(self, target_dnn_output):
        '''
        Often times, you want to process the output of your target dnn into something nicer.
        This function is called everytime a target dnn output is computed and allows you to process it.
        If it is not defined, it will simply return the input.
        '''
        return target_dnn_output

    def do_mining(self):
        '''
        The mining step of constructing a TASTI. We will use an embedding dnn to compute embeddings
        of the entire dataset. Then, we will use FPFRandomBucketter to choose "distinct" datapoints
        that can be useful for triplet training.
        '''
        if self.config.do_mining:
            model = self.get_pretrained_embedding_dnn()
            try:
                model.cuda()
                # model.cpu()
                model.eval()
            except:
                pass
            # 将当前视频所有图片用预训练好的dnn inferance
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.get_num_workers(),
                pin_memory=True
            )

            embeddings = []
            with torch.no_grad():

                for batch in tqdm(dataloader, desc='Embedding DNN ： iterate the images to generate embeddings'):
                    batch = batch.cuda()
                    # device = torch.device("cpu")  # 将张量移动到CPU上
                    # batch = batch.to(device)
                    output = model(batch).cpu()
                    embeddings.append(output)
                embeddings = torch.cat(embeddings, dim=0)
                embeddings = embeddings.numpy()

            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_train, self.seed)
            reps, _, _ = bucketter.bucket(embeddings, self.config.max_k)
            self.training_idxs = reps
        else:
            print('number of training instances: ', self.config.nb_train)
            self.training_idxs = self.rand.choice(
                len(self.get_embedding_dnn_dataset(train_or_test='train')),
                size=self.config.nb_train,
                # replace=False
                replace=True
            )

    def do_training(self):
        '''
        通过三重态损失对嵌入dnn进行微调。
        Fine-tuning the embedding dnn via triplet loss. 
        '''
        if self.config.do_training:
            model = self.get_target_dnn()
            model.eval()
            # model.cpu()
            model.cuda()

            for idx in tqdm(self.training_idxs, desc='Target DNN'):
                self.target_dnn_cache[idx]
            # print('[target_dnn_cache]: ',self.target_dnn_cache[144])
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            triplet_dataset = tasti.data.TripletDataset(
                dataset=dataset,
                target_dnn_cache=self.target_dnn_cache,
                list_of_idxs=self.training_idxs,
                is_close_fn=self.is_close,
                index=self,
                length=self.config.nb_training_its
            )
            dataloader = torch.utils.data.DataLoader(
                triplet_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.get_num_workers(),
                pin_memory=True
            )
            print('[First_FPF_reps_training_data]: ', self.training_idxs)
            model = self.get_embedding_dnn()
            model.train()
            model.cuda()
            # model.cpu()
            loss_fn = tasti.TripletLoss(self.config.train_margin)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)

            # device = torch.device("cpu")
            for anchor, positive, negative in tqdm(dataloader, desc='Training Step'):
                anchor = anchor.cuda(non_blocking=True)
                positive = positive.cuda(non_blocking=True)
                negative = negative.cuda(non_blocking=True)
                # anchor = anchor.to(device)  # 将anchor移动到设备（CPU或其他）
                # positive = positive.to(device)  # 将positive移动到设备（CPU或其他）
                # negative = negative.to(device)  # 将negative移动到设备（CPU或其他）

                e_a = model(anchor)
                e_p = model(positive)
                e_n = model(negative)

                optimizer.zero_grad()
                loss = loss_fn(e_a, e_p, e_n)
                loss.backward()
                optimizer.step()

            save_directory = os.path.join(self.cache_dir, 'model.pt')
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            print('[Here detect]')
            torch.save(model.state_dict(), save_directory)
            self.embedding_dnn = model
        else:
            print("here")
            # else 代码块：
            # 如果do_training 为假，则执行以下代码块。这里直接将
            # embedding_dnn属性设置为预训练的嵌入DNN模型，而不进行训练。
            # 然后，清理或重新初始化目标DNN模型的缓存，并更新为测试集的缓存。
            flag = True
            if flag:
                self.embedding_dnn = self.get_pretrained_embedding_dnn()
                print('[unload model.pt , so it is Tasti-pt]')
            else:
                self.embedding_dnn = self.load_model()
                print('[load model.pt , so it is Tasti]')
        # 正在清理或重新初始化该缓存
        del self.target_dnn_cache
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='test'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='test')

    def do_infer(self):
        '''
        With our fine-tuned embedding dnn, we now compute embeddings for the entire dataset.
        '''
        save_directory = os.path.join(self.cache_dir, f'embeddings_{self.config.video_name}.npy')
        if self.config.do_infer:
            print('[exe_infer]')
            model = self.embedding_dnn
            model.eval()
            model.cuda()
            dataset = self.get_embedding_dnn_dataset(train_or_test='test')
            # print("[dataset]: ", dataset)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.get_num_workers(),
                pin_memory=True
            )

            embeddings = []
            for batch in tqdm(dataloader, desc='Inference'):
                try:
                    batch = batch.cuda()
                    # device = torch.device("cpu")  # 将张量移动到CPU上
                    # batch = batch.to(device)
                except:
                    pass
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()
            # for embedding in embeddings:
            #     print('[embedding]: ', embedding)
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_directory), exist_ok=True)
            np.save(save_directory, embeddings)
            self.embeddings = embeddings
        else:
            try:
                self.embeddings = np.load(save_directory)
            except:
                self.embeddings = None

    def do_bucketting(self, percent_fpf=0.75):
        '''
        Given our embeddings, cluster them and store the reps, topk_reps, and topk_dists to finalize our TASTI.
        '''
        if self.embeddings is None:
            raise ValueError()

        if self.config.do_bucketting:
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_buckets, self.seed)
            # print("[embeddings]: ", self.embeddings)
            self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k,
                                                                          percent_fpf=percent_fpf)
            # print("[Tasti_pi reps]:", self.reps)
            # print("[Tasti_pi len(reps)]:", len(self.reps))
            # print("[Tasti_pi topk_reps]:", self.topk_reps)
            # print("[Tasti_pi topk_dists]:", self.topk_dists)
            np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
            np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
            np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)
        else:
            self.reps = np.load(os.path.join(self.cache_dir, '/reps.npy'))
            self.topk_reps = np.load(os.path.join(self.cache_dir, '/topk_reps.npy'))
            self.topk_dists = np.load(os.path.join(self.cache_dir, '/topk_dists.npy'))

    def crack(self):
        cache = self.target_dnn_cache.cache
        cached_idxs = []
        for idx in range(len(cache)):
            if cache[idx] != None:
                cached_idxs.append(idx)
        cached_idxs = np.array(cached_idxs)
        bucketter = tasti.bucketters.CrackingBucketter(self.config.nb_buckets)
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k, cached_idxs)

        np.save(os.path.join(self.cache_dir, '/reps.npy'), self.reps)
        np.save(os.path.join(self.cache_dir, '/topk_reps.npy'), self.topk_reps)
        np.save(os.path.join(self.cache_dir, '/topk_dists.npy'), self.topk_dists)

    def init(self, percent_fpf=0.75):
        print('index initializing....')
        self.do_mining()
        print('mining complete!')
        self.do_training()
        print('training complete!')
        self.do_infer()
        print('inferring complete!')
        self.do_bucketting(percent_fpf=percent_fpf)
        print('bucketing complete!')
        # 1.2 在所有抽取的代表帧上运行oracle模型
        # print(self.reps)
        # for rep in tqdm(self.reps, desc='Target DNN Invocations'):
        #     self.target_dnn_cache[rep]
        k = 0
        len_reps = len(self.reps)
        channel_layer = get_channel_layer()
        update_step_length = int(float(len_reps) * 0.05)
        async_to_sync(channel_layer.group_send)(
            "algorithm_type_group",
            {
                "type": "algorithm.update",
                "algorithm_type": 'index_construct',
                "consumer_type": 'algorithm_type'
            }
        )
        for rep in tqdm(self.reps, desc='generate index_cache'):
            self.run_oracle_model(rep)
            if k % 10 == 0 or k == len_reps -1:
                progress = (k + 1) / len_reps * 100
                # print('[progress]: ', progress)
                async_to_sync(channel_layer.group_send)(
                    "progress_group",
                    {
                        "type": "progress.update",
                        "progress": progress,
                        "algorithm_type": 'index_construct'
                    }
                )
            k += 1

    # def run_oracle_model(self,rep):
    #     print('running oracle ws')
    def run_oracle_model(self, rep):
        print('[[running oracle index old111]] ')
        if (rep) in self.index_cache:
            # print('[run here]')
            return self.index_cache[rep]
        else:
            result = self.model(self.images[rep])
            print('[Index_result]: ', result)
            # Iterate through detections for each image in the batch
            detections = result.xyxy[0]
            if len(detections) > 0:
                # Iterate through detections for this image
                for detection in detections:
                    # Extract bounding box coordinates and confidence score
                    bbox = detection[:4]  # First four elements are usually bbox coordinates
                    confidence = detection[4]  # Fifth element is often the confidence score
                    object_name = self.model.names[int(detection[5])]  # Get class name using model.names
                    if object_name == 'car' and confidence.item() >= 0.01:
                        # Create the inner dictionary
                        inner_dict = {
                            'object_name': object_name,
                            'confidence': confidence.item(),  # Convert tensor to float
                            'xmin': int(bbox[0]),
                            'ymin': int(bbox[1]),
                            'xmax': int(bbox[2]),
                            'ymax': int(bbox[3])
                        }
                        # print('[inner_dict]: ', inner_dict)
                        # Add the inner dictionary to your index_cache
                        if rep in self.index_cache:
                            self.index_cache[rep].append(inner_dict)
                        else:
                            self.index_cache[rep] = [inner_dict]
            if rep not in self.index_cache:
                self.index_cache[rep] = [{
                    'object_name': None,
                    'confidence': -1,  # Convert tensor to float
                    'xmin': 0,
                    'ymin': 0,
                    'xmax': 0,
                    'ymax': 0
                }]
                # print('[No target Object]: ')
                # print('[here]')
            return self.index_cache[rep]

    def run_oracle_model_full(self, rep):
        # print('[run_full]')
        if rep in self.full_index_cache:
            # print('[cache_hit]')
            return self.full_index_cache[rep]
        else:
            with torch.no_grad():
                result = self.model(self.images[rep])
                # result = result.to('cpu')
                # Iterate through detections for each image in the batch
                detections = result.xyxy[0]
                if len(detections) > 0:
                    # Iterate through detections for this image
                    for detection in detections:
                        # Extract bounding box coordinates and confidence score
                        bbox = detection[:4]  # First four elements are usually bbox coordinates
                        confidence = detection[4]  # Fifth element is often the confidence score
                        object_name = self.model.names[int(detection[5])]  # Get class name using model.names
                        if (object_name == 'car'):
                            # Create the inner dictionary
                            inner_dict = {
                                'object_name': object_name,
                                'confidence': confidence.item(),  # Convert tensor to float
                                'xmin': int(bbox[0]),
                                'ymin': int(bbox[1]),
                                'xmax': int(bbox[2]),
                                'ymax': int(bbox[3])
                            }
                            # print('[inner_dict]: ', inner_dict)
                            # Add the inner dictionary to your index_cache
                            if rep in self.full_index_cache:
                                self.full_index_cache[rep].append(inner_dict)
                            else:
                                self.full_index_cache[rep] = [inner_dict]
                    if rep not in self.full_index_cache:
                        self.full_index_cache[rep] = [{
                            'object_name': None,
                            'confidence': -1,  # Convert tensor to float
                            'xmin': 0,
                            'ymin': 0,
                            'xmax': 0,
                            'ymax': 0
                        }]
                    # print(self.full_index_cache)
                    return self.full_index_cache[rep]
                else:
                    self.index_cache[rep] = [{
                        'object_name': None,
                        'confidence': -1,  # Convert tensor to float
                        'xmin': 0,
                        'ymin': 0,
                        'xmax': 0,
                        'ymax': 0
                    }]
                    return self.index_cache[rep]

    def read_or_write_cache(self):
        pass

    def read_csv_to_index(self, filepath):
        """
        Reads a CSV file and stores the data in an index dictionary.

        Args:
            filepath (str): The path to the CSV file.

        Returns:
            dict: The index dictionary with numerical keys and inner dictionaries as values.
        """
        print("[File Path]: ", filepath)
        # Check if the file exists, if not, create it
        if not os.path.exists(filepath):
            # If the directory doesn't exist, create it
            output_dir = os.path.dirname(filepath)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Create an empty CSV file
            with open(filepath, 'w', newline=''):
                pass

        index = {}
        # Open the CSV file for reading
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Iterate through each row in the CSV file
            for row in reader:
                # Extract the 'ind' value as the key
                key = int(row['frame'])
                # Create an inner dictionary with the remaining data
                inner_dict = {
                    'object_name': row['object_name'],
                    'confidence': float(row['confidence']),
                    'xmin': int(row['xmin']),
                    'ymin': int(row['ymin']),
                    'xmax': int(row['xmax']),
                    'ymax': int(row['ymax'])
                }
                # Add the inner dictionary to the index with the key
                if key in index:
                    index[key].append(inner_dict)
                else:
                    index[key] = [inner_dict]
        return index

    # def load_model(self, model_path='D:/Projects/PyhtonProjects/thesis/tasti_data/cache/tasti_triplet/model.pt'):
    def load_model(self, model_path='/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/tasti_triplet/model.pt'):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 128)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 设置模型为推理模式
        return model
