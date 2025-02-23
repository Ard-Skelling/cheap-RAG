# -*- coding: utf-8 -*-
import os
import io
import re
import json
import fitz
import shutil
import logging
import traceback
import subprocess
import pandas as pd
from docx import Document

__all__ = ['ParseFile']

"""
    下面解析结构都是以迭代器返回, 原因是考虑到以后如果要支持(目前前端是控制在50M) 10G, 100G, 1T的知识库的时候, 一次性加载到内存不现实, 故需要分批读取
"""

class ParseFile(object):

    def __init__(self) -> None:
        super(ParseFile, self).__init__()

    @staticmethod
    def get_docx_iters(bytes_data, suffix=".doc", csid=""):
        """解析docx结果"""
        try:
            document_text = ""
            if suffix == ".doc":
                try:
                    # step1: 先创建唯一标识文件名
                    curr_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    folder_name = csid
                    folder_path = os.path.join(curr_path, "tempfiles/{}".format(folder_name))
                    print("----------folder_path--------", folder_path)
                    # step2: 将doc的bytes数据写入到文件夹中
                    os.makedirs(folder_path)
                    doc_file_path = os.path.join(folder_path, folder_name + ".doc")
                    with open(doc_file_path, "wb") as f_w:
                        f_w.write(bytes_data)
                    f_w.close()
                    # step3: 用libreoffice将doc转成docx
                    cmd = "soffice --headless --convert-to docx --outdir {} {}".format(folder_path, doc_file_path).split()
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, timeout=300)
                    logging.info("Sucessed to doc convert docx")
                    docx_file_path = os.path.join(folder_path, folder_name + ".docx")
                    # step4: 读取转换成功的docx bytes数据
                    with open(docx_file_path, "rb") as f_r:
                        bytes_data = f_r.read()
                    f_r.close()
                    # step5: 删除唯一标识创建的临时文件夹
                    shutil.rmtree(folder_path)
                    logging.info("detecte folder: {} sucessful".format(folder_path))
                except Exception as err:
                    logging.error("faild to doc convert docx")
                    logging.error(traceback.format_exc())

            document = Document(io.BytesIO(bytes_data))
            for paragraph in document.paragraphs:
                text = re.sub(r'\s', '', paragraph.text)
                if text:
                    document_text += text
            return document_text
        except Exception as err:
            logging.error("Faild to parse docx file.")
            logging.error(traceback.format_exc())
        
    @staticmethod
    def get_txt_iters(bytes_data):
        """解析txt结果"""
        try:
            txt_str = bytes_data.decode('utf-8')
            for line in txt_str.splitlines():
                yield line
        except Exception as err:
            logging.error("Faild to parse txt file.")
            logging.error(traceback.format_exc())
            
    @staticmethod
    def get_pdf_iters(bytes_data):
        """解析pdf结果"""
        try:
            document_text = ""
            data = io.BytesIO(bytes_data)
            doc = fitz.open(stream=data, filetype="pdf")
            for page in doc:
                blocks = page.get_text('dict').get('blocks')
                for block in blocks:
                    if block['type'] != 0:
                        continue
                    lines = block.get('lines', [])
                    for line in lines:
                        for span in line.get("spans"):
                            document_text += span.get('text', '')
            return document_text
        except Exception as err:
            logging.error("Faild to parse pdf file.")
            logging.error(traceback.format_exc())
    
    @staticmethod
    def get_excel_iters(bytes_data, suffix=".xlsx"):
        """解析excel类型文件"""
        try:
            stream = io.BytesIO(bytes_data)
            if suffix in [".csv"]:
                dfs = pd.read_csv (stream, sheet_name=None, header=0)
            else:
                dfs = pd.read_excel(stream, sheet_name=None, header=0)

            for sheet_name, df in dfs.items():
                logging.info("Sheet Name: {}".format(sheet_name))
                for _, row in df.iterrows():
                    yield row.values
        except Exception as err:
            logging.error("Faild to parse xlsx file.")
            logging.error(traceback.format_exc())

    @staticmethod
    def get_json_iters(bytes_data):
        """解析json, 以迭代器返回"""
        try:
            json_str = bytes_data.decode('utf-8')
            for line in json_str.splitlines():
                yield json.loads(line)
        except Exception as err:
            logging.error("Faild to parse json file.")
            logging.error(traceback.format_exc())





if __name__=='__main__':
    # file_path = r'./问答对模板.xls'
    file_path = r'./test.json'

    with open(file_path, 'rb') as f:
        bytes_data = f.read()
        for value in ParseFile().get_json_result(bytes_data):
            print("-----------value-----------", value)
