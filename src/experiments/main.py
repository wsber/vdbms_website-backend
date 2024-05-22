from src.system_architecture.alternate import EKO_alternate
from src.system_architecture.parameter_search import EKOPSConfig, EKO_PS
from src.system_architecture.ratio import EKO_ratio
from src.system_architecture.mab import EKO_mab
import time
import os

from benchmarks.stanford.tasti.tasti.seiden.queries.queries import NightStreetAggregateQuery, \
    NightStreetAveragePositionAggregateQuery, \
    NightStreetSUPGPrecisionQuery, \
    NightStreetSUPGRecallQuery


def execute_ekoalt_rq3(images, video_name, anchor_percentage=0.5, nb_buckets=10000):
    ekoconfig = EKOPSConfig(video_name, nb_buckets=nb_buckets)
    ekoalt = EKO_alternate(ekoconfig, images, initial_anchor=anchor_percentage)
    ekoalt.init()

    return ekoalt


def execute_ekoalt_rq4(images, video_name, anchor_percentage=0.5, nb_buckets=10000, exploit_ratio=0.5):
    ekoconfig = EKOPSConfig(video_name, nb_buckets=nb_buckets)
    ekoalt = EKO_ratio(ekoconfig, images, initial_anchor=anchor_percentage, exploit_ratio=exploit_ratio)
    ekoalt.init()

    return ekoalt


def execute_ekomab(images, video_name, keep=False,
                   category='car', nb_buckets=10000,
                   anchor_percentage=0.4, c_param=2,
                   confidence_threshold=0.95, error_threshold=0.1,
                   reacall_threshold=0.95, precision_threshold=0.95,
                   label_propagate_al=0):
    ekoconfig = EKOPSConfig(video_name,
                            category=category, nb_buckets=nb_buckets,
                            user_confidence_threshold=confidence_threshold, user_error_threshold=error_threshold,
                            reacall_threshold=reacall_threshold, precision_threshold=precision_threshold,
                            label_propagate_al=label_propagate_al)
    # base = '/home/wangshuo_20/pythonpr/thesis_data/video_data'
    base = '/home/wangshuo_20/pythonpr/VDBMS_ws/media'
    # directory = os.path.join(base, video_name, 'video.mp4')
    directory = os.path.join(base, video_name)
    print(directory)
    ekomab = EKO_mab(ekoconfig, images, directory, c_param=c_param, anchor_percentage=anchor_percentage, keep=keep)
    ekomab.init()

    return ekomab


THROUGHPUT = 1 / 140


def query_process_aggregate(index, error=0.05, y=None, images=None):
    # if index is None:
    #     assert(y is not None)
    # if y is None:
    #     assert(index is not None)

    st = time.perf_counter()
    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=error, confidence=0.05, y=y, images=images)
    nb_samples = result['nb_samples']

    et = time.perf_counter()
    t = et - st

    return t, result


def query_process_precision(index, dnn_invocation=1000, y=None, images=None):
    if index is None:
        assert (y is not None)
    if y is None:
        assert (index is not None)
    times = []
    st = time.perf_counter()
    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(dnn_invocation, y=y, images=images)
    et = time.perf_counter()
    times.append(et - st)
    # precision = result['precision']
    # recall = result['recall']

    return times, result


def get_labels(index, dnn_invocation=1000, y=None):
    if index is None:
        assert (y is not None)
    if y is None:
        assert (index is not None)
    query = NightStreetSUPGRecallQuery(index)
    result = query.execute_metrics(dnn_invocation, y=y)
    precision = result['precision']
    recall = result['recall']
    return query


def query_process_recall(index, dnn_invocation=1000, y=None, images=None):
    if index is None:
        assert (y is not None)
    if y is None:
        assert (index is not None)

    query = NightStreetSUPGRecallQuery(index)
    print('[recall_dnn_invocation]: ', dnn_invocation)
    result = query.execute_metrics(dnn_invocation, y=y, images=images)
    # precision = 'test'
    # recall = 'test'
    # precision = result['precision']
    # recall = result['recall']
    # print('precision:',precision,'recall',recall)
    return result
