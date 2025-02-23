# -*- coding: utf-8 -*-
import os
import re
import json
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor
from configs.vector_database_config import GLOBAL_CONFIG


summry="""你是一个文章总结专家，擅长提炼文章的主旨和关键信息，能够迅速识别并总结文章的核心论点和结论，现在给你一篇文章进行总结，要求如下：
        1.快速浏览文章，确定文章的主题和结构。
        2.识别文章中的关键段落和句子，提炼出主要观点。
        3.将关键信息整合成连贯的总结，确保包含文章的核心论点和结论。
        4.只返回总结内容
        5.例子：'总结：本文介绍了智能家居、人工智能、无人驾驶、虚拟现实和5G通信等前沿科技领域的现状和发展趋势。智能家居通过物联网技术实现了家电的远程控制和自动化管理，未来将成为每个家庭的标准配置。人工智能在多个行业广泛应用，提高了工作效率并创造了新的服务模式。无人驾驶汽车依靠先进的传感器和算法，在复杂交通环境中安全行驶，预计未来几年内将进入市场。虚拟现实技术提供了沉浸式体验，应用场景将越来越广泛，如远程医疗、虚拟旅游等。5G通信技术的推广将极大提升网络速度和连接稳定性，促进物联网的发展，实现万物互联。科技的发展为未来带来了无限的可能，一个更加智能、便捷、高效的社会正向我们走来。'
        现在给你文章：{}\n"""
        


# table_prompt = """你是一位专注于数据分析和Markdown格式的专家，擅长从表格数据中提取关键信息，并能够根据这些信息生成一系列详细的问题。请根据Markdown格式的表格数据，从中抽取出多个具体的、针对性强的问题。

# 要求：
# 1.输出的问题应直接针对表格中的具体内容，避免过于宽泛的问题；
# 2.问题应涉及表格各列中的具体细节，如数值、名称、日期、机构、内容等；
# 3.可以针对表格内容中的每一个具体数值、目标或关键词提问。例如可以遍历表格的所有行列，对每个单元格提出问题；
# 4.当提供了表格标题时，问题应该与表格的完整标题相结合，例如：表3人员工资分配中的人员工资等级III的基本工资是多少？；
# 5.生成的问题应涵盖表格的所有列和行，确保全面性；
# 6.生成少量跨列或跨行的综合性问题，体现表格数据之间的关联；
# 7.请尽可能的生成多一些问题。
# 8.直接按照以下格式输出：
# Q1：[具体问题1]
# Q2：[具体问题2]
# Q3：[具体问题3]
# ...

# 表格标题：{}

# 表格内容：
# {}

# 请严格按照要求对表格进行全面问题抽取，确保问题数量充足且涵盖表格的各个方面。"""


table_prompt = """你是一位专注于数据分析和Markdown格式的专家，擅长从表格数据中提取关键信息，并能够根据这些信息生成一系列详细的问题。请根据Markdown格式的表格数据，从中抽取出多个具体的、针对性强的问题。

要求：
1. 生成的问题应涵盖表格的所有列和行，确保全面性；
2. 生成跨列或跨行的综合性问题，体现表格数据之间的关联；
3. 问题的数量控制在2个;
4. 直接按照以下格式输出：
Q1：[具体问题1]
Q2：[具体问题2]

表格标题：{}

表格内容：
{}

请严格按照要求对表格进行全面问题抽取，确保问题质量充足且涵盖表格的各个方面。"""



# text_prompt = """请根据文本内容，从中抽取出一些具体、针对性强的问题。

# 要求：
# 1.输出的问题应直接针对文本的具体内容，避免过于宽泛的问题
# 2.问题应涉及表格各列中的具体细节等信息
# 3.直接按照以下格式输出：
# Q1：[具体问题1]
# Q2：[具体问题2]
# Q3：[具体问题3]
# ...

# 文本内容：
# {}

# 请严格遵守以上要求，对文本进行问题抽取。"""


text_prompt = """请根据文本内容，从中抽取出一些具体、针对性强的问题。

要求：
1. 输出的问题应直接针对文本的具体内容，避免过于宽泛
2. 问题应对内容信息进行高度概括，把握本质信息
3. 问题的数量控制在2个
4. 直接按照以下格式输出：
Q1：[具体问题1]
Q2：[具体问题2]

文本内容：
{}

请严格遵守以上要求，对文本进行问题抽取。"""



class Inference(object):
    """"""

    def __init__(self, config):

        super(Inference, self).__init__()
        self.ocr_path = config.get("ocr_path", "")
        assert os.path.exists(self.ocr_path), "ocr result not exists."
        self.block_thread_pool = ThreadPoolExecutor(max_workers=32)  # 针对每个需要生成问答对的block块的线程池
        self.block_inner_thread_pool = ThreadPoolExecutor(max_workers=5)  # 每个块内文本和表格, 图片的线程池
        self._init_ocr_result()

        self.address = GLOBAL_CONFIG.ocr_config['url']
        
    def _init_ocr_result(self):
        
        file_name = os.path.basename(self.ocr_path)
        self.image_folder_path = os.path.join(self.ocr_path, "images")  # 文件的相关图片路径
        self.json_path = os.path.join(self.ocr_path, "mds")  # 表格的json文件, 后续是markdown

        self.md_result_path = os.path.join(self.ocr_path, file_name + ".md")

    def read_json_file(self, type="md"):
        """
            type: chooce list: ["content", "middle", "model"]
        """
        try:
            assert type in ["content", "middle", "model"], "file type not support."
            with open(eval("self.{}_result_path"), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as err:
            print(traceback.format_exc())
        finally:
            f.close()
    

    def postprocess_md(self):
        """
            预处理md, 将一些断裂的合在一行
            思路: ocr出来的md结果有一些一句话断开了, 所以做了一些合并, 并且在一个# 标题下面的内容区域和区域用一个空行区分,  不同title 之间用两个空行区分, 表格或者图片也做了严格区分
        """
        try:
            self.post_md_result_path =  os.path.splitext(self.md_result_path)[0] + "_post.md"
            fw = open(self.post_md_result_path, "w")
        
            with open(self.md_result_path, "r", encoding="utf-8") as f:
                words = ""
                pre_line = ""
                flag = False
                for line in f:
                    line = line.strip()
                    if re.search(r"^#(.*)", line):  # 判断是否以# 开头的标题
                        words = words.strip()
                        if words:
                            flag = True
                            words_list = words.strip().split("\n\n")
                            prefix_words = ""
                            for word in words_list:
                                if not word :
                                    continue
                                if re.search(r"(\!\[\]\(images|\!\[\]\(mds)", word):
                                    if prefix_words:
                                        fw.write(prefix_words + "\n\n")
                                        prefix_words = ""
                                    fw.write(word + "\n\n")
                                    continue
                            
                                if not re.search(r"[。？]", word[-1]):
                                    prefix_words += word
                                else:
                                    prefix_words += word
                                    fw.write(prefix_words + "\n\n")
                                    prefix_words = ""

                            if prefix_words:   
                                fw.write(prefix_words + "\n\n")
                                prefix_words = ""

                            words = ""
                        
                        if flag:
                            fw.write("\n" + line + "\n")
                        else:
                            fw.write(line + "\n")
                        flag=False

                    else:
                        if line:
                            pre_line += line + "\n" 
                        else:
                            words += pre_line + "\n"
                            pre_line = ""

                # 当没有标题的情况
                if words or pre_line:
                    fw.write(words + pre_line)
                    words = ""
                    pre_line = ""
        except Exception as err:
            print(traceback.format_exc())
        # finally:
        #     f.close()
        #     fw.close()

    def send_llm_requests(self, content):
        """调用大模型"""

        params = {
            "model": GLOBAL_CONFIG.llm_config['model'],
            "messages": [
                {
                    "role": "user",
                    "content": content[:25000]
                }
            ],
            "max_tokens": 8000,
            "presence_penalty": 1.05,
            # "frequency_penalty": 1.0,
            # "seed": 52,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False
        }
        headers = {"Content-type": "application/json"}
        url = GLOBAL_CONFIG.llm_config['url']
        response = requests.post(url=url, json=params, headers=headers, timeout=GLOBAL_CONFIG.request_timeout)
        return response

        
    def process_block(self, parent_title, title, words_list, block_index, file_name, is_enhanced=False, assigned_type='text'):
        """处理单个block块内的问答"""
        def nonsense_regex(text, is_title=False):
            """处理文本中的无意义字符。Milvus会将某些字符识别为不止一个字符"""
            if is_title:
                text = re.sub('[^0-9a-zA-Z一-龥 \.·,，。]', '', text)
            text = re.sub(r'··+', '··', text)
            text = re.sub(r'。。+', '。。', text)
            text = re.sub(r'\.\.+', '..', text)
            text = re.sub(r' +', ' ', text)
            return text.strip()

        block_future_list = []
        # 分别为文本和表格生成问题
        new_words = ""
        for i, words in enumerate(words_list):
            # 提取表格链接
            md_reg = re.search(r"mds/[\w./-]+\.md", words)
            # 注意: 由于当前OCR会同时返回图片的文本和图像链接，且图标题会放到文本中，
            # 因此不对图片pattern做处理
            # jpg_reg = re.search(r"images/[\w./-]+\.jpg", words)
            if md_reg:
                md_url = md_reg.group()    # 形如mds/page1_table0.md或mds/page0_image.md
                # 做成表格图片实际存储位置url
                stem_pattern = r'page\d+_\w+\d+'
                stem = ''.join(re.findall(stem_pattern, md_url))
                image_url = f'images/{stem}.jpg'
                # step1: 读取md内容
                md_path = os.path.join(self.ocr_path, md_url)
                md_content = ""
                if not os.path.exists(md_path):
                    if GLOBAL_CONFIG.debug_mode:
                        raise FileExistsError(f'{md_path} not exist.')
                    else:
                        print(f'{md_path} not exist.')
                        continue
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                # step2: 调用大模型
                # todo 标题是否带上
                if 'table' in words:
                    content = table_prompt.format(title, md_content)
                    future = self.block_inner_thread_pool.submit(self.send_llm_requests, content)
                    block_future_list.append(("table", (future, md_content, image_url)))
                elif 'image' in words:
                    # 处理OC从图片中识别到的文本，使用上下文和OCR文本，调用文本模板构造Q&A。
                    image_question = words.replace(md_url, "").replace("![]()", "").strip()
                    md_content = f'**图片描述**：\n{image_question}\n\n**图片内容**：\n{md_content}'
                    # 处理image上下文为空的情况，使用当前章节作为上下文
                    if not md_content.strip():
                        md_content = '\n\n'.join(words_list)
                    content = text_prompt.format(md_content)
                    future = self.block_inner_thread_pool.submit(self.send_llm_requests, content)
                    block_future_list.append(("image", (future, md_content, image_url)))
                else:
                    raise ValueError(f'OCR markdown result must be table or image, but get {md_url}')
                continue
            
            # 提取图片链接
            # 注意: 由于当前OCR会同时返回图片的文本和图像链接，且图标题会放到文本pattern(形如)中，
            # 因此不对图片pattern做处理
            # if jpg_reg:
            #     image_url = jpg_reg.group()    # 形如images/page0_image.jpg
            #     image_path = os.path.join(self.ocr_path, image_url)
            #     # image_path = self.address + image_url
            #     image_question = words.replace(image_url, "").replace("![]()", "").strip()
            
            #     # 后续调用多模态大模型
            #     block_future_list.append(("image", [image_question, "", image_url]))
                # continue

            # 处理文本
            new_words += words + "\n"
        
        result_list = []
        # 最后处理文本 todo 这里后续还要判断文本长度, 标题是否带上
        if is_enhanced:
            content = text_prompt.format(new_words)
            future = self.block_inner_thread_pool.submit(self.send_llm_requests,  content)
            block_future_list.append((assigned_type, (future, new_words, "")))
        else:
            result_list.append({
                "question": '',
                "answer": nonsense_regex(new_words),
                "type": assigned_type,
                "url": '',
                # 由于milvus数据库的bug，会将某些字符识别为不止1个字符，导致文本长度超限，需要特殊处理
                "parent_title": nonsense_regex(parent_title, is_title=True)[:100],
                "title": nonsense_regex(title, is_title=True)[:100],
                "index": block_index,
                "file_name": file_name
            })

        for (type_, futures) in block_future_list:
            future, answer, url = futures[0], futures[1], futures[2]
            
            # 目前是图片
            if isinstance(futures, list):
                result_list.append({
                    "question": future,
                    "answer": nonsense_regex(answer),
                    "url": url,
                    "type": type_,
                    # 由于milvus数据库的bug，会将某些字符识别为不止1个字符，导致文本长度超限，需要特殊处理
                    "parent_title": nonsense_regex(parent_title, is_title=True)[:100],
                    "title": nonsense_regex(title, is_title=True)[:100],
                    "index": block_index,
                    "file_name": file_name
                })
            if isinstance(futures, tuple):
                result = future.result()
                questions="".join(re.findall(r'"content":"(.*?)"', result.text, re.DOTALL))
                question_list = questions.split("\\n")
                for question in question_list:
                    question_rag = re.search(r"Q\d+：(.*)", question)
                    if not question_rag:
                        continue
                    result_list.append({
                        "question": question_rag.group(1),
                        "answer": nonsense_regex(answer),
                        "type": type_,
                        "url": url,
                        # 由于milvus数据库的bug，会将某些字符识别为不止1个字符，导致文本长度超限，需要特殊处理
                        "parent_title": nonsense_regex(parent_title, is_title=True)[:100],
                        "title": nonsense_regex(title, is_title=True)[:100],
                        "index": block_index,
                        "file_name": file_name
                    })
        return result_list

    def split_text(self, file_name, is_enhanced=False):
        """
            目前只处理#和# 之间的文档进行提问, 后续需要ocr对标题进行打分 例如 ### 1级标题  ## 2级标题  # 3级标题
        """
        # step1: 判断是否有多级标题
        # step2: 如果多级标题就把第一个标题置为title
        # file_name = ""
        # base_name = os.path.basename(self.ocr_path)
        # for folder in os.listdir(self.ocr_path):
        #     if folder.startswith(base_name+"_origin"):
        #         file_name = base_name + os.path.splitext(folder)[-1]
        #         break

        def has_many_dots(title):
            crit_0 = title.count('·') >= 5 or title.count('·') / len(title) >= 0.3
            crit_1 = title.count('.') >= 5 or title.count('.') / len(title) >= 0.3
            return crit_0 or crit_1

        with open(self.post_md_result_path, "r", encoding="utf-8") as f:
            block_index = 1
            count = 0
            words_list = []
            parent_title = ""
            title = ""
            future_list = []
            # 归集所有的标题, 生成单独的一个文本块, 存储整篇文章标题结构
            title_list = []
            # 这里通过迭代遍历为了对于很大的文件占用内存
            for line in f:
                if count == 2:
                    # 判断是否是2级标题
                    if re.search(r"^#", words_list[0]) and re.search(r"^#", words_list[1]):
                        parent_title = words_list.pop(0)
                        title = words_list.pop(0)
                        title_list.extend([parent_title, title])
                    elif re.search(r"^#", words_list[0]):
                        title = words_list.pop(0)
                        title_list.append(title)
                    else:
                        title = ""
                    new_words_list = "".join(words_list).strip().split("\n\n")
                    future = self.block_thread_pool.submit(self.process_block, parent_title, title, new_words_list, block_index, file_name, is_enhanced)
                    future_list.append(future)

                    words_list = []
                    block_index += 1
                # 版面片段之间有两行的间隔
                if not line.strip():
                    count += 1
                else:
                    count = 0
                words_list.append(line)
            
            # 当没有标题的情况
            if words_list:
                new_words_list = "".join(words_list).strip().split("\n\n")
                future = self.block_thread_pool.submit(self.process_block, parent_title, title, new_words_list, 0, file_name, is_enhanced)
                future_list.append(future)

            # 处理归集的标题
            # 用······的数量和占比过滤掉目录部分
            title_list = [t.strip() for t in title_list if (t.strip('#').strip() and not has_many_dots(t))]
            if title_list:
                future = self.block_thread_pool.submit(self.process_block, '', '', title_list, 0, file_name, is_enhanced, assigned_type='outline')
                future_list.append(future)
            
            # fw = open("./rag_slice_result_0830.json", "w", encoding="utf-8")
            results = []
            for future in future_list:
                result = future.result()
                for res in result:
                    results.append(res)
                    # json.dump(res, fw, ensure_ascii=False)
                    # fw.write("\n")
            return results
    
    def predict(self, file_name, is_enhanced=False):
        """
            目前只处理以#为主的, 因为没有拿到没有#的情况  到时候要具体分析
        """
        try:
            # step1: 读取ocr返回的md文件, 并与处理
            self.postprocess_md()
            # step2: 切割md返回的结果, 大模型生成问题构造qa, 输出最终的问答对
            results = self.split_text(file_name, is_enhanced)
            return results
        except Exception as err:
            raise err
            print(traceback.format_exc())

if __name__=="__main__":
    config = {
        # "ocr_path": "./11《南方电网公司“十四五”工业互联网技术专项规划》（征求意见稿）"
        "ocr_path": "./Niha"
    }
    

    infer = Inference(config)

    infer.predict("Niha.docx")