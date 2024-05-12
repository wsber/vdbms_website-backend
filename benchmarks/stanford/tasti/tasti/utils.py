import tasti
import numpy as np
import torch, torchvision



"""
Both BlazeIt and SUPG assume for the sake of fast experiments that you have access to all of the Target DNN outputs.
These classes will allow you to still use the BlazeIt and SUPG algorithms by executing the Target DNN in realtime.
"""


# 定义了一个名为 DNNOutputCache 的类，用于缓存目标DNN模型的输出结果。
class DNNOutputCache:
    """初始化方法，用于创建DNNOutputCache对象。
    参数：
        target_dnn：目标DNN模型。
        dataset：用于输入目标DNN模型的数据集。
        target_dnn_callback：目标DNN模型输出的回调函数，默认为恒等函数（即不做任何处理）。
    方法内部：
        将目标DNN模型移动到GPU并设置为评估模式。
        初始化缓存列表，其长度与数据集长度相同，每个元素初始化为None。
        初始化计数器nb_of_invocations，用于记录调用次数。
    """
    def __init__(self, target_dnn, dataset, target_dnn_callback=lambda x: x):
        target_dnn.cuda()
        target_dnn.eval()
        self.target_dnn = target_dnn
        self.dataset = dataset
        self.target_dnn_callback = target_dnn_callback
        self.length = len(dataset)
        self.cache = [None]*self.length
        self.nb_of_invocations = 0
        
    def __len__(self):
        return self.length
    """
        实现了对象的索引访问。
        如果缓存中不存在索引为idx的结果，则进行以下操作：
            从数据集中获取索引为idx的数据，并转换为张量。
            将数据传递给目标DNN模型进行前向传播，并经过target_dnn_callback处理。
            将处理后的结果存储到缓存中。
            更新调用次数。
        返回索引为idx的结果。
    """
    def __getitem__(self, idx):
        if self.cache[idx] == None:
            with torch.no_grad():
                record = self.dataset[idx].unsqueeze(0).cuda()
                result = self.target_dnn(record)
            result = self.target_dnn_callback(result)
            self.cache[idx] = result
            self.nb_of_invocations += 1
        return self.cache[idx]
            
class DNNOutputCacheFloat:
    # wser add here
    def __init__(self, target_dnn_cache, scoring_fn, idx,model,images=None,index=None):
        self.target_dnn_cache = target_dnn_cache
        self.scoring_fn = scoring_fn
        self.idx = idx
        self.images = images
        self.index_cache = index.index_cache
        self.index = index
        def override_arithmetic_operator(name):
            def func(self, *args):
                # print(self.idx)
                # print('[call this override_arithmetic_operator_def to construct?]')
                # results = model(self.images[self.idx], size=640)
                # results = self.index.run_oracle_model_full(self.idx)
                results = self.index.run_oracle_model(self.idx)
                # max_confidence = self.get_confidence_of_yolvo5s_result(results)
                # value = max_confidence
                value = self.scoring_fn(results)
                value = np.float32(value)
                args_f = []
                for arg in args:
                    if type(arg) is tasti.utils.DNNOutputCacheFloat:
                        arg = np.float32(arg)
                    args_f.append(arg)
                value = getattr(value, name)(*args_f)
                return value 
            return func
        
        operator_names = [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__", 
            "__neg__", 
            "__pos__", 
            "__radd__",
            "__rmul__",
        ]
            
        for name in operator_names:
            setattr(DNNOutputCacheFloat, name, override_arithmetic_operator(name))
        
    def __repr__(self):
        return f'DNNOutputCacheFloat(idx={self.idx})'
    
    def __float__(self):
        # value = self.target_dnn_cache[self.idx]
        # value = self.get_confidence_of_yolvo5s_result(self.index_cache[self.idx])
        # results = self.index.run_oracle_model(self.idx)
        # max_confidence = self.get_confidence_of_yolvo5s_result(results)
        # value = max_confidence
        # value = self.scoring_fn(results)
        # print('[call this __float__ to transfer]')
        results = self.index.run_oracle_model(self.idx)
        value = self.scoring_fn(results)
        return float(value)

    def get_confidence_of_yolvo5s_result(self, results):
        # print('[results]: ',results)
        max_person_confidence = 0.0
        # 遍历检测结果中的每个预测框
        for row in results:
            # 如果预测框的类别是 person
            if row['object_name'] == 'car':  # person 类别的索引为 0
                # 更新最高置信度的值
                max_person_confidence = max(max_person_confidence, row['confidence'])
        # 输出最高置信度的 person 类别预测框的置信度
        # print("最高置信度的 car 类别预测框的置信度:", max_person_confidence)
        return max_person_confidence