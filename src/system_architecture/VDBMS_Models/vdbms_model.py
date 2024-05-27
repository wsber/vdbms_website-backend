from src.motivation.main import load_dataset


class VDBMSModel:
    def __init__(self, model_name, config):
        self.name = model_name
        self.config = config

    def load_images_to_Model(self, video_name):
        images = load_dataset(video_name)
        return images

    def collect_running_data(self, collect_datas):
        print(collect_datas)

    def exe_pre_query(self):
        pass

    def exe_recall_query(self):
        pass

    def exe_aggregate_query(self):
        pass


class SeidenModel(VDBMSModel):
    def __init__(self, images, config, model_name):
        self.images = images
        super().__init__(model_name, config)

    def collect_running_data(self, collect_datas):
        print(collect_datas)

    def exe_pre_query(self):
        pass

    def exe_recall_query(self):
        pass

    def exe_aggregate_query(self):
        pass


class TastiModel(VDBMSModel):
    def __init__(self, images, config, model_name):
        self.images = images
        super().__init__(model_name, config)

    def collect_running_data(self, collect_datas):
        print(collect_datas)

    def exe_pre_query(self):
        pass

    def exe_recall_query(self):
        pass

    def exe_aggregate_query(self):
        pass


class SVMModel(VDBMSModel):
    def __init__(self, images, config, model_name):
        self.images = images
        super().__init__(model_name, config)

    def collect_running_data(self, collect_datas):
        print(collect_datas)

    def exe_pre_query(self):
        pass

    def exe_recall_query(self):
        pass

    def exe_aggregate_query(self):
        pass


class SeidenModelExtra(VDBMSModel):
    def __init__(self, images, config, model_name):
        self.images = images
        self.seiden_model_pointer = None
        super().__init__(model_name, config)

    def set_seiden_model_pointer(self, seiden_model):
        self.seiden_model_pointer = seiden_model

    def collect_running_data(self, collect_datas):
        print(collect_datas)

    def exe_pre_query(self):
        pass

    def exe_recall_query(self):
        pass

    def exe_aggregate_query(self):
        pass
