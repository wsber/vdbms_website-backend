# Generated by Django 3.2 on 2024-05-08 03:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('seiden', '0005_vdbmsmodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='vdbmsmodel',
            name='parameter_url',
            field=models.URLField(default='D:/projects', verbose_name='超参数加载地址'),
            preserve_default=False,
        ),
    ]