from src.system_architecture.alternate import EKO_alternate
from src.system_architecture.parameter_search import EKOPSConfig, EKO_PS
from src.system_architecture.ratio import EKO_ratio
from src.system_architecture.mab import EKO_mab
import time
import os
### import the queries
from benchmarks.stanford.tasti.tasti.seiden.queries.queries import NightStreetAggregateQuery, \
    NightStreetAveragePositionAggregateQuery, \
    NightStreetSUPGPrecisionQuery, \
    NightStreetSUPGRecallQuery
def get_index_construction_time(images, video_name, keep = False,  category = 'car', nb_buckets = 10000, anchor_percentage = 0.2, c_param = 2):
    ekoconfig = EKOPSConfig(video_name, category = category, nb_buckets=nb_buckets)
    # base = 'D:/Projects/PyhtonProjects/thesis/video_data'
    base = '/home/wangshuo_20/pythonpr/thesis_data/video_data'
    directory = os.path.join(base, video_name)
    print(directory)
    ekomab = EKO_mab(ekoconfig, images, directory, c_param = c_param, anchor_percentage=anchor_percentage, keep = keep)
    ekomab.init()
    query = NightStreetAggregateQuery(ekomab)
    query.finish_index_building()
    return ekomab