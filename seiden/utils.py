# my_project/utils.py

from rest_framework_jwt.settings import api_settings

def my_jwt_response_handler(token, user=None, request=None):
    return {
        'token': token,
        'user': {
            'id': user.id,
            'name': user.username,
            # 添加其他需要返回的用户信息
        }
    }
