# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os


class PublicEngineConfig(object):
    """公共配置"""

    # 推理根目录
    BASE_PATH = os.environ.get("ROOT_PATH") or os.path.join(os.path.dirname(os.getcwd()), "app_milvus_server")

    # 模型存放目录
    BASE_ASSETS_PATH = BASE_PATH + "/assets/"

    # 日志文件路径
    LOGGING_FILE_PATH = os.path.join(BASE_PATH, "logs/app_server.log")

    # 服务版本号
    VERSION = "v1.1_20240603"  # 用来记录发布的代码标识

    # 默认向量数据
    PUB_VEC_DB = "pub_vec_db"