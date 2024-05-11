from django.db import models
from django.contrib.auth.models import AbstractUser


class User(models.Model):
    name = models.CharField(max_length=30, default="unknown", verbose_name="姓名")
    username = models.CharField(max_length=30, verbose_name="账号", unique=True)
    phone = models.CharField(max_length=20, verbose_name="电话")
    password = models.CharField(max_length=100, verbose_name="密码")
    email = models.EmailField(verbose_name="邮箱", max_length=100, null=True, blank=True)
    gender = models.CharField(max_length=5, verbose_name="性别", default="男")
    role = models.CharField(max_length=10, verbose_name="角色", default="USER")
    REQUIRED_FIELDS = ['phone', 'password']
    USERNAME_FIELD = "username"

    class Meta:
        verbose_name = "用户信息"
        verbose_name_plural = verbose_name
        ordering = ["username"]

    def __str__(self):
        return self.name + " with " + self.username


class Video(models.Model):
    name = models.CharField(max_length=100, verbose_name="视频名称")
    description = models.TextField(verbose_name="视频描述", blank=True, null=True)
    duration = models.CharField(max_length=20, verbose_name="视频时长")
    upload_time = models.DateTimeField(auto_now_add=True, verbose_name="上传时间")
    url = models.URLField(max_length=200, verbose_name="视频地址")  # 新增视频地址字段

    class Meta:
        verbose_name = "视频信息"
        verbose_name_plural = "视频信息"
        ordering = ["-upload_time"]  # 根据上传时间降序排列

    def __str__(self):
        return self.name


class VdbmsModel(models.Model):
    name = models.CharField(max_length=100, verbose_name="模型名称")
    description = models.TextField(verbose_name="模型描述", blank=True, null=True)
    recommend = models.IntegerField(verbose_name="模型推荐度")
    upload_time = models.DateTimeField(auto_now_add=True, verbose_name="模型上传时间")
    parameter_url = models.URLField(max_length=200, verbose_name="超参数加载地址")  # 新增视频地址字段
    class Meta:
        verbose_name = "模型信息"
        verbose_name_plural = "模型信息"

    def __str__(self):
        return self.name
