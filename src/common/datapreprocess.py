
# -*- coding: utf-8 -*-
import json
import uuid
import decimal
import datetime
import numpy as np
import hashlib


generate_md5 = lambda text: hashlib.md5(text.encode('utf-8')).hexdigest()


class JSONEncoder(json.JSONEncoder):

    def default(self, o):
        """
        如有其他的需求可直接在下面添加
        :param o:
        :return:
        """
        if isinstance(o, datetime.datetime):
            # 格式化时间
            return o.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(o, datetime.date):
            # 格式化日期
            return o.strftime('%Y-%m-%d')
        if isinstance(o, decimal.Decimal):
            # 格式化高精度数字
            return float(o)
        if isinstance(o, uuid.UUID):
            # 格式化uuid
            return str(o)
        if isinstance(o, set):
            return list(o)
        if isinstance(o, bytes):
            # 格式化字节数据
            return o.decode("utf-8")
        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)