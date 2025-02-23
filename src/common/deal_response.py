# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import json
from flask import Response
from common import JSONEncoder

__all__ = ["response", "BadRequest"]


STATUS_CODES = {
    "200": "success",
    "10001": "The request method is not supported",
    "10002": "Parameter error, please check the request parameters",
    "10003": "Service internal error"
}


def response_func(csid="", code=200, message=None):
    """
        响应码
    """
    result = {
        "code": str(code) if code else "200",
        "message": message if message else STATUS_CODES[str(code)] if code else "success",
        "csid": csid,
        "timestamp": int(time.time())
    }
    return json.dumps(result, ensure_ascii=False, cls=JSONEncoder)

class BadRequest(Exception):
    def __init__(self, message):
        self.message = message


