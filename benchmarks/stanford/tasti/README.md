# Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data

This is the official project page for "Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data"

Please read the [paper](https://arxiv.org/abs/2009.04540) for full technical details.

# Requirements

Install the requitements with `pip install -r requirements.txt`. You will also need (via `pip install -e .`):
- [SWAG](https://github.com/stanford-futuredata/swag-python)
- [BlazeIt](https://github.com/stanford-futuredata/blazeit)
- [SUPG](https://github.com/stanford-futuredata/supg)pip
- Install the tasti package with `pip install -e .` as well.

To reproduce the experiments, your machine will need:
- 300+GB of memory
- 500+GB of space
- GPU (e.g., NVIDIA V100, TITAN V)

On other datasets, hardware requirements will vary.

# Installation
Feel free to replace `conda` with your own installation method.
```
git clone https://github.com/stanford-futuredata/swag-python.git
cd swag-python/
conda install -c conda-forge opencv
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/blazeit.git
cd blazeit/
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge pyclipper
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/supg.git
cd supg/
pip install pandas feather-format
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/tasti.git
cd tasti/
pip install -r requirements.txt
pip install -e .
```

# Reproducing Experiments

We provide code for creating a TASTI for the `night-street` video dataset along with all the queries mentioned in the paper (aggregation, limit, SUPG, position, etc). You can download the `night-street` video dataset [here](https://drive.google.com/drive/folders/1phQuGu4oWwbArurprqruMztTdP1Fzz2F?usp=sharing). Download the `2017-12-14.zip` and `2017-12-17.zip` files. Unzip the files and place the video data in `/lfs/1/jtguibas/data` (feel free to change this path in `night_street_offline.py`). For speed purposes, the target dnn will not run in realtime and we have instead provided the outputs [here](https://drive.google.com/drive/folders/1XKZmBb0AvCBJX11bJGdoxdgMozoiSuWf?usp=sharing). Place the csv files in `/lfs/1/jtguibas/data`. Then, you can reproduce the experiments by running:

```
python tasti/examples/night_street_offline.py
```

We also provide an online version of the code that allows you to run the target dnn in realtime. For efficiency purposes, we use [Mask R-CNN ResNet-50 FPN](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) as the target dnn. However, the actual model used in the paper is the Mask R-CNN X 152 model available in [detectron2](https://github.com/facebookresearch/detectron2). We encourage you to replace the inference with TensorRT or another model serving system for more serious needs.

To run the WikiSQL example, download the data [here](https://github.com/salesforce/WikiSQL) and place `train.jsonl` in `/lfs/1/jtguibas/data` (again, feel free to change this path inside `wikisql_offline.py`).

To run the CommonVoice example, download the data [here](https://commonvoice.mozilla.org/en/datasets) and install `audioset_tagging_cnn` from [here](https://github.com/qiuqiangkong/audioset_tagging_cnn). You will also need to download the `Cnn10` and `ResNet22` models from there. After, adjust the paths in `commonvoice_offline.py` accordingly and make sure to install any missing dependencies.

复制实验

我们提供了为夜街视频数据集创建 TASTI 的代码，以及论文中提到的所有查询（聚合、限制、SUPG、位置等）。您可以在此下载夜街视频数据集。下载 2017-12-14.zip 和 2017-12-17.zip 文件。解压缩后，将视频数据放到 /lfs/1/jtguibas/data（可在 night_street_offline.py 中随意更改路径）。为了提高速度，目标 dnn 不会实时运行，我们在此提供了输出结果。将 csv 文件放到 /lfs/1/jtguibas/data。然后，您就可以运行以下命令来重现实验：

我们还提供了在线版本的代码，让您可以实时运行目标 dnn。为了提高效率，我们使用 Mask R-CNN ResNet-50 FPN 作为目标 dnn。不过，本文中使用的实际模型是 detectron2 中的 Mask R-CNN X 152 模型。我们鼓励您使用 TensorRT 或其他模型服务系统来替代推理，以满足更高的需求。

要运行 WikiSQL 示例，请在此处下载数据，并将 train.jsonl 放在 /lfs/1/jtguibas/data 中（同样，您也可以在 wikisql_offline.py 中更改该路径）。

要运行 CommonVoice 示例，请在此处下载数据，并在此处安装 audioset_tagging_cnn。您还需要从此处下载 Cnn10 和 ResNet22 模型。之后，相应调整 commonvoice_offline.py 中的路径，并确保安装任何缺失的依赖项。

# Customizing TASTI

Our code allows for you to create your own TASTI. You will have to inherit the `tasti.Index` class and implement a few functions:

```python
import tasti

class MyIndex(tasti.Index):
    def is_close(self, a, b):
        '''
        Return a Boolean of whether records a and b are 'close'.
        '''
        raise NotImplementedError

    def get_target_dnn_dataset(self, train_or_test='train'):
        '''
        Return a torch.seiden_utils.data.Dataset object.
        '''
        raise NotImplementedError

    def get_embedding_dnn_dataset(self, train_or_test='train'):
        '''
        Return a torch.seiden_utils.data.Dataset object.
        '''
        raise NotImplementedError

    def get_target_dnn(self):
        '''
        Return a torch.nn.Module object.
        '''
        raise NotImplementedError

    def get_embedding_dnn(self):
        '''
        Return a torch.nn.Module object.
        '''
        raise NotImplementedError

    def get_pretrained_embedding_dnn(self):
        '''
        Optional if do_mining is False.
        Return a torch.nn.Module object.
        '''
        raise NotImplementedError

    def target_dnn_callback(self, target_dnn_output):
        '''
        Optional if you don't want to process the target_dnn_output.
        '''
        return target_dnn_output

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        '''
        Optional if you want to run the target dnn in realtime.
        Allows for you to override the target_dnn_cache when you have the
        target dnn outputs already cached.
        '''
        raise NotImplementedError

class MyQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        '''
        Maps a target_dnn_output into a feature/scalar you are interested in.
        Note that this is an aggregate query, so this query will try to estimate the total sum of these scores.
        '''
        return len(target_dnn_output)

config = tasti.IndexConfig()
config.nb_buckets = 500

index = MyIndex(config)
index.init()

query = MyQuery()
result = query.execute()
print(result)
```

# Config
These are the options available in `tasti.IndexConfig` which get passed into the `tasti.Index` object.
- `do_mining`, Boolean that determines whether the mining step is skipped or not
- `do_training`, Boolean that determines whether the training/fine-tuning step of the embedding dnn is skipped or not
- `do_infer`, Boolean that allows you to either compute embeddings or load them from `./cache`
- `do_bucketting`, Boolean that allows you to compute the buckets or load them from `./cache`
- `batch_size`, general batch size for both the target and embedding dnn
- `train_margin`, controls the margin parameter of the triplet loss
- `max_k`, controls the k parameter described in the paper (for computing distance weighted means and votes)
- `nb_train`, controls how many datapoints are labeled to perform the triplet training
- `nb_buckets`, controls the number of buckets used to construct the index
- `nb_training_its`, controls the number of datapoints are passed through the model during training
