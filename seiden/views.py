# %load_ext autoreload
# %autoreload 2
import hashlib
import os.path
import sys

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework_jwt.settings import api_settings

sys.path.append('/home/wangshuo_20/pythonpr/seiden_ws')
import uuid
from django.contrib.auth.hashers import make_password, check_password
from django.forms import model_to_dict
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from seiden.models import User
from seiden.models import Video
from seiden.models import VdbmsModel
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json
from django.db.models import Q
from src.motivation.main import *
from rest_framework_jwt.views import ObtainJSONWebToken
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework_jwt.settings import api_settings

images = None  # 全局变量，初始化为 None

'''
    User Module
'''


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])  # 允许未经身份验证的访问
def my_obtain_jwt_token(request):
    # 从请求中获取用户名和密码
    username = request.data.get('username')
    password = request.data.get('password')
    query = Q(username__icontains=username)
    # 使用构建好的查询条件进行过滤
    user = User.objects.filter(query).first()
    if user is not None and check_password(password, user.password):
        # 验证成功，生成 JWT
        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
        payload = jwt_payload_handler(user)
        token = jwt_encode_handler(payload)
        # 自定义响应
        return Response({
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'profile_url': user.profile_url,
                'phone': user.phone,
                'role': user.role
                # 添加其他需要返回的用户信息
            }
        }, status=status.HTTP_200_OK)
    else:
        # 验证失败，返回错误响应
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


# 引入json模块 JsonResponse
@csrf_exempt
def login(request):
    data = json.loads(request.body.decode('utf-8'))
    print(data)
    username = data['username']
    password = data['password']
    query = Q(username__icontains=username)
    # 使用构建好的查询条件进行过滤
    obj_users = User.objects.filter(query).values()
    user_obj = obj_users[0]
    if check_password(password, user_obj['password']):
        return JsonResponse({
            'code': 1,
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


@csrf_exempt
def get_query_users(requset):
    # 传递过来的查询条件 -- axios默认是json
    data = json.loads(requset.body.decode('utf-8'))
    print(data)
    try:
        query = Q()
        # 对每个查询条件进行模糊查询，并添加到 Q 对象中
        if 'userName' in data:
            query &= Q(name__icontains=data['userName'])
        if 'userPhone' in data:
            query &= Q(description__icontains=data['userPhone'])
        if 'userRole' in data:
            query &= Q(duration__icontains=data['userRole'])
        # 使用构建好的查询条件进行过滤
        obj_users = User.objects.filter(query).values()
        # 把结果转为list
        obj_users = list(obj_users)
        print(obj_users)
        return JsonResponse({
            'code': 1,
            'data': obj_users
        })
    except Exception as e:
        return JsonResponse({'code': 0, 'msg': '获取视频信息出现异常' + str(e)})


@csrf_exempt
def add_user(request):
    data = json.loads(request.body.decode('utf-8'))
    # 对密码进行哈希加密
    hashed_password = make_password(data['password'])
    print(data)
    try:
        object_user = User(name=data['name'], username=data['username'], profile_url=data['profile_url'],
                           email=data['email'], role=data['role'], phone=data['phone'], password=hashed_password)
        object_user.save()
        obj_students = User.objects.all().values()
        # 把结果转为list
        students = list(obj_students)
        return JsonResponse({
            'code': 1,
            'data': students
        })
    except Exception as e:
        return JsonResponse({'code': -1, 'msg': '失败' + str(e)})


@csrf_exempt
def update_user_profile_url(request):
    data = json.loads(request.body.decode('utf-8'))
    user_id = data['user_id']
    new_profile_url = data['profile_url']
    user = User.objects.get(id=user_id)
    user.profile_url = new_profile_url
    user.save()
    return JsonResponse({
        'code': 1,
        'mag': '头像更新成功'
    })


@csrf_exempt
def delete_user_by_id(request):
    data = json.loads(request.body.decode('utf-8'))
    user_id = data['id']
    try:
        user = User.objects.get(pk=user_id)  # 获取指定 ID 的视频对象
        user.delete()  # 真删除
        return JsonResponse({
            'code': 1,
            'msg': '删除视频成功'
        })
    except Video.DoesNotExist:
        return JsonResponse({
            'code': -1,
            'msg': '删除视频失败，视频不存在'
        })


@csrf_exempt
def delete_batch_users(request):
    data = json.loads(request.body.decode('utf-8'))
    user_ids = data.get('ids', [])  # 获取视频 ID 列表
    # 检查是否提供了视频 ID 列表
    if not user_ids:
        return JsonResponse({
            'code': -1,
            'msg': '请提供视频 ID 列表'
        })
    try:
        # 批量删除视频
        User.objects.filter(pk__in=user_ids).delete()
        return JsonResponse({
            'code': 1,
            'msg': '批量删除成功'
        })
    except Exception as e:
        return JsonResponse({
            'code': -1,
            'msg': '批量删除失败：' + str(e)
        })


'''
    Video Module
'''


@csrf_exempt
def get_video(request, video_id):
    # 获取视频文件路径
    video_path = os.path.join('/home/wangshuo_20/pythonpr/VDBMS_ws/media',
                              '4d63ea26233716e25480b0bac2b12003' + '.mp4')  # 替换为你的实际路径

    # 检查文件是否存在
    if not os.path.exists(video_path):
        return HttpResponse("Video not found", status=404)

    # 获取文件大小
    file_size = os.path.getsize(video_path)

    # 设置响应头信息
    response = StreamingHttpResponse(content_type='video/mp4')
    response['Accept-Ranges'] = 'bytes'  # 支持断点续传
    response['Content-Length'] = file_size  # 设置文件大小

    # 处理请求范围（断点续传）
    if request.headers.get('range'):
        range_header = request.headers.get('range')
        print(range_header)
        if 'bytes=' in range_header:
            if range_header == 'bytes=0-':
                # 读取整个文件
                def file_iterator(file_path):
                    with open(file_path, 'rb') as f:
                        yield f.read()

                response.streaming_content = file_iterator(video_path)
            else :
                start, end = map(int, range_header.split('=')[1].split('-'))

                # 处理范围错误
                if start < 0 or end > file_size:
                    return HttpResponse("Invalid range", status=416)

                # 读取文件部分内容
                def file_iterator(file_path, start, end):
                    with open(file_path, 'rb') as f:
                        f.seek(start)
                        yield f.read(end - start + 1)

                response.streaming_content = file_iterator(video_path, start, end)
                response.status_code = 206  # Partial Content
        else:
            # 返回错误信息，例如 400 Bad Request
            return HttpResponse("Invalid Range header", status=400)
    else:
        # 没有范围请求，读取整个文件
        def file_iterator(file_path):
            with open(file_path, 'rb') as f:
                yield f.read()

        response.streaming_content = file_iterator(video_path)

    return response


@csrf_exempt
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
                             duration=data['duration'], upload_time=data['upload_time'],uuid_name=data['uuid_name'])
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
def delete_video_by_id(request):
    data = json.loads(request.body.decode('utf-8'))
    video_id = data['id']
    try:
        video = Video.objects.get(pk=video_id)  # 获取指定 ID 的视频对象
        video.delete()  # 真删除
        return JsonResponse({
            'code': 1,
            'msg': '删除视频成功'
        })
    except Video.DoesNotExist:
        return JsonResponse({
            'code': -1,
            'msg': '删除视频失败，视频不存在'
        })


@csrf_exempt
def delete_batch_videos(request):
    data = json.loads(request.body.decode('utf-8'))
    video_ids = data.get('ids', [])  # 获取视频 ID 列表
    # 检查是否提供了视频 ID 列表
    if not video_ids:
        return JsonResponse({
            'code': -1,
            'msg': '请提供视频 ID 列表'
        })
    try:
        # 批量删除视频
        Video.objects.filter(pk__in=video_ids).delete()
        return JsonResponse({
            'code': 1,
            'msg': '批量删除成功'
        })
    except Exception as e:
        return JsonResponse({
            'code': -1,
            'msg': '批量删除失败：' + str(e)
        })


'''
    Model Module
'''


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
    data = json.loads(request.body.decode('utf-8'))
    video_name = 'cherry_5min'
    # video_name = data['videos'][0]['name']
    print('[video_name]: ', video_name)
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
    frame_sql = data['frame_sql']
    selfParameters = data['selfParameters']
    print('[frame_sql]: ', frame_sql)
    print('[selfParameters]: ', selfParameters)
    anchor_count = int(len_images * 0.1)
    # try:
    eko = execute_ekomab(images, video_name, nb_buckets=anchor_count)
    print(anchor_count)
    times, result = query_process_precision(eko, dnn_invocation=anchor_count, images=images)
    pre_result = {'inds': [int(ind) for ind in result['inds']], 'y_pred': [float(pred) for pred in result['y_pred']]}
    # 遍历字典的键值对
    # for key, value in pre_result.items():
    #     # 使用 type() 函数获取数据项的类型
    #     value_type = type(value)
    #     # 输出键值对的键、值和数据类型
    #     print(f"Key: {key}, Value: {value}, Type: {value_type}")
    # 将字典 result 转换为 JSON 字符串
    result_json = json.dumps(pre_result)
    print('[Here has executed the pre al]')
    return JsonResponse({
        'code': 1,
        'data': result_json,
        'msg': 'exe pre complete successfully'
    })
    # except Exception as e:
    #     return JsonResponse({'code': -1, 'msg': '失败' + str(e)})


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
        pre_result = {'inds': [int(ind) for ind in result['inds']],
                      'y_pred': [float(pred) for pred in result['y_pred']]}
        result_json = json.dumps(pre_result)
        return JsonResponse({
            'code': 1,
            'data': result_json,
            'msg': 'exe rec complete successfully'
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
        result_json = json.dumps(result)
        return JsonResponse({
            'code': 1,
            'data': result_json,
            'msg': 'exe agg successfully'
        })
    except Exception as e:
        return JsonResponse({'code': -1, 'msg': '失败' + str(e)})
