# -*- coding: utf-8 -*-

from .get_logger import log_file
from .datapreprocess import JSONEncoder, generate_md5
from .deal_response import response_func, BadRequest
from .cost_time import CostTime, cost_time
# from .api_predict_common import VecDataApiCommon
from .api_predict_common_latest import VecDataApiCommon
from .https import send_request_server, send_request_embed_server, send_request_rerank_server