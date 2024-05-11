# %load_ext autoreload
# %autoreload 2
import hashlib
import os.path
import sys

sys.path.append('/home/wangshuo_20/pythonpr/seiden_ws')
import uuid
from django.contrib.auth.hashers import make_password
from django.forms import model_to_dict
from django.http import JsonResponse
from django.shortcuts import render
from django.shortcuts import HttpResponse
# Create your views here.
from seiden.models import User
from seiden.models import Video
from seiden.models import VdbmsModel
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json
from django.db.models import Q
from src.motivation.main import *

images = None  # 全局变量，初始化为 None


# 引入json模块 JsonResponse
@csrf_exempt
def login(request):
    data = json.loads(request.body.decode('utf-8'))
    username = data['username']
    password = data['password']
    query = Q(username__icontains=username)
    # 使用构建好的查询条件进行过滤
    obj_users = User.objects.filter(query).values()
    user_obj = obj_users[0]
    if user_obj.check_password(password):
        return JsonResponse({
            'code': 200,
            'data': user_obj
        })
    else:
        return JsonResponse({'code': -1, 'msg': '登录失败'})


@csrf_exempt
def register(request):
    data = json.loads(request.body.decode('utf-8'))
    username = data['username']
    password = data['password']
    # 对密码进行哈希加密
    hashed_password = make_password(password)
    object_video = User(username=username, password=hashed_password, role='USER')
    object_video.save()
    return JsonResponse({
        'code': 200,
    })


def get_users(request):
    try:
        obj_students = User.objects.all().order_by('id').values()
        # 把结果转为list
        students = list(obj_students)
        return JsonResponse({
            'code': 1,
            'data': students
        })
    except Exception as e:
        return JsonResponse({'code': 0, 'msg': '获取用户信息出现异常' + str(e)})


def get_videos(request):
    try:
        obj_students = Video.objects.all().order_by('id').values()
        # 把结果转为list
        students = list(obj_students)
        return JsonResponse({
            'code': 1,
            'data': students
        })
    except Exception as e:
        return JsonResponse({'code': 0, 'msg': '获取视频信息出现异常' + str(e)})


@csrf_exempt
def get_query_videos(requset):
    # 传递过来的查询条件 -- axios默认是json
    data = json.loads(requset.body.decode('utf-8'))
    print(data)
    try:
        query = Q()
        # 对每个查询条件进行模糊查询，并添加到 Q 对象中
        if 'name' in data:
            query &= Q(name__icontains=data['name'])
        if 'videoDesc' in data:
            query &= Q(description__icontains=data['videoDesc'])
        if 'videoDuration' in data:
            query &= Q(duration__icontains=data['videoDuration'])
        # 使用构建好的查询条件进行过滤
        obj_videos = Video.objects.filter(query).values()
        # 把结果转为list
        obj_videos = list(obj_videos)
        print(obj_videos)
        return JsonResponse({
            'code': 1,
            'data': obj_videos
        })
    except Exception as e:
        return JsonResponse({'code': 0, 'msg': '获取视频信息出现异常' + str(e)})


@csrf_exempt
def upload_video(request):
    """接收上传的文件"""
    rev_file = request.FILES.get('file_data')
    if not rev_file:
        return JsonResponse({'code': 0, 'msg': '文件不存在'})
    # 获得唯一的名字：uuid +hash
    else:
        new_name = get_random_str()
        file_extension = os.path.splitext(rev_file.name)[1].lower()  # 获取文件后缀
        file_path = os.path.join(settings.MEDIA_ROOT, new_name + file_extension)
        print(file_path)
        # 开始写入磁盘
        try:
            f = open(file_path, 'wb')
            # 文件比较大时，分多次写入
            for i in rev_file.chunks():
                f.write(i)
            # 关闭
            f.close()
            return JsonResponse({'code': 1, 'name': new_name + file_extension})
        except Exception as e:
            return JsonResponse({'code': 0, 'msg': str(e)})


@csrf_exempt
def add_video(request):
    data = json.loads(request.body.decode('utf-8'))
    print(data)
    try:
        object_video = Video(name=data['name'], description=data['description'], url=data['url'],
                             duration=data['duration'], upload_time=data['upload_time'])
        object_video.save()
        obj_students = Video.objects.all().values()
        # 把结果转为list
        students = list(obj_students)
        return JsonResponse({
            'code': 1,
            'data': students
        })
    except Exception as e:
        return JsonResponse({'code': -1, 'msg': '失败' + str(e)})


def get_random_str():
    uuid_value = uuid.uuid4()
    uuid_str = str(uuid_value).encode('utf-8')
    md5 = hashlib.md5()
    md5.update(uuid_str)
    return md5.hexdigest()


@csrf_exempt
def get_video_by_id(request):
    """
    根据视频ID获取视频信息
    """
    data = json.loads(request.body.decode('utf-8'))
    video_id = data['id']
    try:
        video = Video.objects.get(id=video_id)
        # 将模型实例转换为字典
        video_data = model_to_dict(video)
        return JsonResponse({
            'code': 1,
            'data': video_data
        })
    except Video.DoesNotExist:
        return JsonResponse({'code': 0, 'msg': '视频不存在'})
    except Exception as e:
        return JsonResponse({'code': 0, 'msg': '获取视频信息出现异常' + str(e)})


@csrf_exempt
def get_videos_by_ids(request):
    """
    根据视频ID获取视频信息
    """
    data = json.loads(request.body.decode('utf-8'))
    id_list = data['ids']
    videos = Video.objects.filter(id__in=id_list).values()
    videos = list(videos)
    return JsonResponse({
        'code': 1,
        'data': videos
    })


@csrf_exempt
def get_models(request):
    try:
        obj_models = VdbmsModel.objects.all().order_by('id').values()
        # 把结果转为list
        obj_models = list(obj_models)
        return JsonResponse({
            'code': 1,
            'data': obj_models
        })
    except Exception as e:
        return JsonResponse({'code': 0, 'msg': '获取模型信息出现异常' + str(e)})


from src.experiments.main import execute_ekomab


@csrf_exempt
def load_data(request):
    global images
    video_name = 'cherry_1min'
    images = load_dataset(video_name)
    # anchor_count = int(len(images) * 0.1)
    # eko = execute_ekomab(images, video_name, nb_buckets=anchor_count)
    # print(anchor_count)
    # times, result = query_process_precision(eko, dnn_invocation=anchor_count, images=images)
    return JsonResponse({'code': 1,
                         'data': {
                             'len_images': len(images),
                             'video_name': video_name
                         }})


@csrf_exempt
def exe_model_pre(request):
    from src.experiments.main import query_process_precision
    global images
    data = json.loads(request.body.decode('utf-8'))
    len_images = data['len_images']
    video_name = data['video_name']
    parameter_list = data['parameter_list']
    print('[parameter_list]: ', parameter_list)
    anchor_count = int(len_images * 0.1)
    try:
        eko = execute_ekomab(images, video_name, nb_buckets=anchor_count)
        print(anchor_count)
        times, result = query_process_precision(eko, dnn_invocation=anchor_count, images=images)
        result['inds'] = list(result['inds'].astype(int))
        result['y_pred'] = list(result['y_pred'].astype(float))
        # 将字典 result 转换为 JSON 字符串
        result_json = json.dumps(result)
        return JsonResponse({
            'code': 1,
            # 'data': list(result['inds'])
            'data': 'exe pre complete successfully'
        })
    except Exception as e:
        return JsonResponse({'code': -1, 'msg': '失败' + str(e)})


@csrf_exempt
def exe_model_rec(request):
    from src.experiments.main import query_process_recall
    global images
    data = json.loads(request.body.decode('utf-8'))
    len_images = data['len_images']
    video_name = data['video_name']
    anchor_count = int(len_images * 0.1)
    try:
        eko = execute_ekomab(images, video_name, nb_buckets=anchor_count)
        print(anchor_count)
        times, result = query_process_recall(eko, dnn_invocation=anchor_count, images=images)
        return JsonResponse({
            'code': 1,
            'data': result
        })
    except Exception as e:
        return JsonResponse({'code': -1, 'msg': '失败' + str(e)})


@csrf_exempt
def exe_model_agg(request):
    from src.experiments.main import query_process_aggregate
    global images
    data = json.loads(request.body.decode('utf-8'))
    len_images = data['len_images']
    video_name = data['video_name']
    anchor_count = int(len_images * 0.1)
    try:
        eko = execute_ekomab(images, video_name, nb_buckets=anchor_count)
        print(anchor_count)
        times, result = query_process_aggregate(eko, images=images)
        return JsonResponse({
            'code': 1,
            'data': result
        })
    except Exception as e:
        return JsonResponse({'code': -1, 'msg': '失败' + str(e)})
