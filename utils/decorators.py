# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月5日
@author: Leo
"""
import os
from functools import wraps


def check_path(path: str):
    """
    检查文件夹是否存在的装饰器
    :param path: 路径
    """
    def _check_path(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(path) is not True:
                os.mkdir(path=path)
            func(*args, **kwargs)
        return wrapper
    return _check_path
