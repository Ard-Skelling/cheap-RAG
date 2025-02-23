# -*- coding: utf-8 -*-
import io
import os
import time
import fitz
import logging
import traceback
import subprocess
import shutil
import requests
import re
import base64
import json
import chardet
import pandas as pd
from docx import Document
from pathlib import Path
from module.parsers.postprocess import Inference
from configs.vector_database_config import GLOBAL_CONFIG, BASEPATH, LLM_HOST, LLM_PORT
from module.parsers.postprocess import summry
from common.https import send_request_server


class ParseFile(object):

    def __init__(self):
        self.llmurl = f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions"
        pass

    @staticmethod
    def clean_format_coding():
        pass
    
    def get_document_result(self, bytes_data,domain, file_name, csid="", is_enhanced=False):
        try:
            # step1: 先创建唯一标识文件名
            curr_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            folder_name = csid
            p_file_name = Path(file_name)
            prefix, suffix = p_file_name.stem, p_file_name.suffix
            suffix = suffix.lower()

            folder_path = os.path.join(curr_path, "tempfiles/{}".format(folder_name))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if suffix == ".doc":
                # step2: 将doc的bytes数据写入到文件夹中               
                doc_file_path = os.path.join(folder_path, prefix + ".doc")
                with open(doc_file_path, "wb") as f_w:
                    f_w.write(bytes_data)
                f_w.close()
                # step3: 用libreoffice将doc转成docx
                # cmd = "soffice --headless --convert-to docx --outdir {} {}".format(folder_path, doc_file_path).split()
                cmd = ['soffice', '--headless', '--convert-to', 'docx', '--outdir', folder_path, doc_file_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, timeout=300)
                logging.info("Sucessed to doc convert docx")
                docx_file_path = os.path.join(folder_path, prefix + ".docx")

                with open(docx_file_path, "wb") as f_w:
                    f_w.write(bytes_data)
                f_w.close()

                # cmd = "soffice --headless --convert-to pdf --outdir {} {}".format(folder_path, docx_file_path).split()
                cmd = ['soffice', '--headless', '--convert-to', 'pdf', '--outdir', folder_path, doc_file_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, timeout=300)
                logging.info("Sucessed to convert docx to pdf")
                pdf_file_path = os.path.join(folder_path, prefix + ".pdf")

                with open(pdf_file_path, "rb") as f:
                    pdf_bytes = f.read()
            elif suffix == ".docx":
                docx_file_path = os.path.join(folder_path, prefix + ".docx")
                with open(docx_file_path, "wb") as f_w:
                    f_w.write(bytes_data)
                f_w.close()

                # step4: 用libreoffice将docx转成pdf
                # cmd = "soffice --headless --convert-to pdf --outdir {} {}".format(folder_path, docx_file_path).split()
                cmd = ['soffice', '--headless', '--convert-to', 'pdf', '--outdir', folder_path, docx_file_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, timeout=300)
                logging.info("Sucessed to convert docx to pdf")
                pdf_file_path = os.path.join(folder_path, prefix + ".pdf")

                with open(pdf_file_path, "rb") as f:
                    pdf_bytes = f.read()
            elif suffix == ".pdf":
                pdf_bytes = bytes_data
            ocr_start_time = time.time()
            # 定义OCR请求的URL
            url = GLOBAL_CONFIG.ocr_config['url']
            # 使用minio后的OCR接口，启用时间2024-12-06
            if GLOBAL_CONFIG.ocr_config['use_minio']:
                document = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_name = f'{prefix}.pdf'
                body = {"collection_name":domain, "file": document, "file_name": pdf_name}
                response = requests.post(url, json=body, timeout=GLOBAL_CONFIG.ocr_config['timeout'])
            # 旧OCR接口
            else:
                files = [("files", (prefix + ".pdf", pdf_bytes, "application/pdf"))]
                response = requests.post(url, files=files, timeout=GLOBAL_CONFIG.ocr_config['timeout'])
            result = json.loads(response.text)
            file_n=prefix+".md"
            content = result[file_n]

            # 获取文本总结语块，在send_request_server中按最长25000字截断
            zj = summry.format(content)
            summary = send_request_server(self.llmurl, zj)

            logging.info("ocr cost time: {} s".format(time.time() - ocr_start_time))
            gen_question_start_time = time.time()
            save_folder_path = os.path.join(folder_path, prefix)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)

            for key, value in result.items():
                if re.findall(r'^page\d+_\w+\d+\.md', key):
                    json_folder_path = os.path.join(save_folder_path, "mds")
                    if not os.path.exists(json_folder_path):
                        os.makedirs(json_folder_path)
                    with open(os.path.join(json_folder_path, key), "w", encoding="utf-8") as f:
                        f.write(value)
                else:
                    with open(os.path.join(save_folder_path, key), "w", encoding="utf-8") as f:
                        f.write(value)
            config = {
                "ocr_path": save_folder_path
            }
            self.post_infer = Inference(config)
            results = self.post_infer.predict(file_name, is_enhanced)
            summary_dict = json.loads(summary)
            future={'question': '', 'answer': summary_dict['choices'][0]['message']['content'], 'type': 'summary', 'url': '', 'parent_title': '', 'title': '', 'index': 0, 'file_name': file_name}
            results.append(future)
            logging.info("llm gen question cost time: {}".format(time.time() - gen_question_start_time))
            # 存储文档解析结果为中间产物
            if GLOBAL_CONFIG.debug_mode:
                ocrs_dir = BASEPATH.joinpath('.ocrs')
                ocrs_dir.mkdir(parents=True, exist_ok=True)
                with open(str(ocrs_dir.joinpath("{}.json".format(file_name))), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)

            return results

        except Exception as err:
            # print(err)
            logging.error(traceback.format_exc())
        finally:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)

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
                    logging.info("----------folder_path--------", folder_path)
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
                text = re.sub(r"\s", "", paragraph.text)
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
                df = pd.read_csv(stream, header=0)
                for _, row in df.iterrows():
                    yield row.values
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
            if re.findall(r'^\s*{\s*"question"', json_str):
                for line in json_str.splitlines():
                    yield json.loads(line)
            elif re.findall(r'^\s*\[\s*{', json_str):
                for ele in json.loads(json_str):
                    yield ele
            else:
                data = json.loads(json_str)
                data = data['qa_pairs']
                for ele in data:
                    yield ele
        except Exception as err:
            logging.error("Faild to parse json file.")
            logging.error(traceback.format_exc())

    @staticmethod
    def get_pdf_result(bytes_data):
        """解析pdf结果"""
        try:
            document_text = ""
            data = io.BytesIO(bytes_data)
            doc = fitz.open(stream=data, filetype="pdf")
            for page in doc:
                blocks = page.get_text("dict").get("blocks")
                for block in blocks:
                    if block["type"] != 0:
                        continue
                    lines = block.get("lines", [])
                    for line in lines:
                        for span in line.get("spans"):
                            document_text += span.get("text", "")
            return document_text
        except Exception as err:
            logging.error("Faild to parse pdf file.")
            logging.error(traceback.format_exc())
    
    @staticmethod
    def check_encoding(bytes_data):
        """检查编码格式"""
        # 去乱码和纠正编码格式错误
        encode_info = chardet.detect(bytes_data)
        encoding_format = encode_info.get("encoding", "")
        if encoding_format and encode_info["encoding"] != "utf-8":
            return bytes_data.decode(encode_info["encoding"]).encode("utf-8")
        else:
            return bytes_data     

    def get_json_result(self, bytes_data):
        """
            解析json 结果
            后续建议做成迭代器的方式
        """
        try:
            bytes_data = self.check_encoding(bytes_data)
            data = json.loads(bytes_data.decode("utf-8"))
            results = []
            for line in data:
                if "question" not in line or "answer" not in line:
                    continue
                results.append(line)
            
            return results
        except Exception as err:
            logging.error(err)

        # data = io.BytesIO(bytes_data)
        # 迭代打印, 防止太大一次性内存损耗太大
        # for line in ijson.items(data, "item"):
        #     if "question" not in line or "answer" not in line:
        #         continue
        #     yield line

    
    def get_xlsx_result(self, bytes_data, suffix):
        try:
            
            bytes_data = self.check_encoding(bytes_data)
            results = []
            if suffix.lower() == ".xlsx":
                df = pd.read_excel(io.BytesIO(bytes_data), engine="openpyxl", usecols=[0, 1], header=None)
            elif suffix.lower() == ".xls":
                df = pd.read_excel(io.BytesIO(bytes_data), engine="xlrd", usecols=[0, 1], header=None)
            elif suffix.lower() == ".csv":
                df = pd.read_csv(io.BytesIO(bytes_data), usecols=[0, 1], encoding="utf-8", header=None)
            else:
                raise ValueError("Unsupported file format: {}".format(suffix))
            
            # 构造成字典列表
            for index, row in df.iterrows():
                if index==0 and (row[0] in ["question", "query", "问题"] and row[1] in ["answer", "答案"]):
                    continue
                results.append({
                    "question": row[0],
                    "answer": row[1]
                })
            return results

        except Exception as err:
            logging.error(err)

    
    def get_txt_result(self, bytes_data, max_length=1000):
        """
            max_length: 做成超参
            思路: 1. 判断一行文字是否低于30个字, ,如果低于30个字 则认为是小标题, 然么对应的标题之间的进行合并为一段, 连续两个低于30个字符则不合并
                  2. 对于长度超过5000的文本进行切割, 切成5000字以内的长度
                  
        """
        try:
            def split_long_text():
                """切割长文本"""
                # 用正则表达式匹配句号和问号
                pattern = r'(?<=[。!?])\s*'
                current_segment = ""
                for sentence in re.split(pattern, words):
                    if len(current_segment) + len(sentence) + 1 <= max_length:
                        current_segment += sentence + ' '
                    else:
                        if current_segment:
                            text_list.append(current_segment.strip())
                        current_segment = sentence + ' '

                # 添加最后一个片段
                if current_segment:
                    text_list.append(current_segment.strip())

            bytes_data = self.check_encoding(bytes_data)
            data = io.BytesIO(bytes_data)
            
            is_title = False
            text_list = []
            words = ""
            for row in data:
                row = row.decode("utf-8").strip()
                if len(row) <= 30 and not is_title:
                    is_title = True
                    if words:
                       # 太长要进行截断
                        if len(words) > max_length:
                            split_long_text()
                        else:
                            text_list.append(words)
                    words = row + "\n"
                else:
                    words += row + "\n"
                    is_title = False
            if words:
                split_long_text()
            
            results = []
            for text in text_list:
                results.append({
                    "question": "",
                    "answer": text
                })
            return results
        except Exception as err:
            logging.error(err)


if __name__=="__main__":
    ...