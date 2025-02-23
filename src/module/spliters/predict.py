# -*- coding: utf-8 -*-
import re
import logging
import traceback


class SplitterInference(object):
    """动态切割文本"""
    def __init__(self, split_config):
        
        self.max_length = split_config.get('max_length', 300)
        self.min_count = split_config.get('min_count', 20)

    def predict(self, text_list):
        """"""
        try:
            split_sentence_list = []
            if isinstance(text_list, str):
                text_list = [text_list]
                
            for text in text_list:
                sentence_list = re.split(r'[!?。！？\?]', text)
                if len(sentence_list) == 1:
                    for i in range(0, len(sentence_list[0]), self.max_length - 1):
                        sentence = sentence_list[0][i:i + self.max_length - 1]
                        if len(sentence) < self.min_count or not text:
                            continue
                        if re.search(r'[\,\.\。\?\!\@\#\$\%\^\&\*\)\）\}\|\”]', sentence[-1]):
                            sentence = sentence[:-1] + '。'
                        else:
                            sentence = sentence + "。"
                    
                        split_sentence_list.append(sentence)
                    return split_sentence_list

                merge_sentence_list = []
                for sentence in sentence_list:  
                    if not sentence:
                        continue

                    if len(sentence) + len("".join(merge_sentence_list)) < self.max_length:
                        merge_sentence_list.append(sentence + '。')
                        continue
                    
                    if len(sentence) < self.max_length:
                        split_sentence_list.append("".join(merge_sentence_list))
                        merge_sentence_list = [sentence + '。']
                    else:
                        split_sentence_list.append("".join(merge_sentence_list))
                        merge_sentence_list = []

                        for i in range(0, len(sentence), self.max_length - 1):
                            text = sentence[i:i + self.max_length - 1]
                            if len(text) < self.min_count or not text:
                                continue

                            if re.search(r'[\,\.\。\?\!\@\#\$\%\^\&\*\)\）\}\|\”]', text[-1]):
                                text = text[:-1] + '。'
                            else:
                                text = text + "。"

                            if (i + self.max_length - 1) > len(text):
                                merge_sentence_list = [text]
                            else:
                                merge_sentence_list.append(text)
                if merge_sentence_list:
                    split_sentence_list.append("".join(merge_sentence_list))
                
            return split_sentence_list
        except Exception as err:
            logging.error(traceback.format_exc())
            logging.error('text spliter faild.')
            return []
        
if __name__=='__main__':

    import os
    config = {
        "max_length": 250,
        "min_count": 20
    }
    infer = SplitterInference(config)

    # folder_path = r''
    # for root, dirs, files in os.walk(folder_path):
    #     if files:
            
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             words = ""
    #             with open(file_path, 'r', encoding='utf-8') as f:
    text = "直升机巡检数据处理业务指南范围本业务指南用于规范110kV及以上电压等级交直流架空输电线路的直升机巡检数据采集及分析工作，无人机巡检数据采集及分析可参考执行。规范性引用文件下列文件中的条款通过本指南的引用而成为本指南的部分条款或内容。凡是注日期的引用文件，其随后所有的修改单（不包括勘误的内容）或修订版均不适用于本指南。凡是不注日期的引用文件，其最新版本适用于本指南。中华人民共和国国务院、中央军事委员会令第371号通用航空飞行管制条例CCAR-91-R2一般运行和飞行规则GB50233110～750kV架空输电线路施工及验收规范GB26859电力安全工作规程（电力线路部分）DL/T288架空输电线路直升机巡视技术导则DL/T289架空输电线路直升机巡视作业标志DL/T436高压直流架空送电线路技术导则DL/T741架空送电线路运行规程DL/T5217220kV～500kV紧凑型架空输电线路设计技术规程DL/T1346直升机激光扫描输电线路作业技术规程CH/T8023机载激光雷达数据处理技术规范Q/CSG210015-2014中国南方电网有限责任公司设备缺陷管理办法术语及定义直升机巡检HelicopterInspection以直升机为平台，搭载红外、紫外、可见光等设备对架空输电线路进行巡查和检测的作业。光电吊舱OptoelectronicPod将可见光相机、可见光摄像机、红外热像仪、紫外成像仪等一种或多种光电传感器集成于陀螺稳定平台，并可挂载于机体下方的密闭化设备，一般为球形或圆柱形。巡检员NavigationInspection直升机巡检过程中使用照相机、稳像仪或光电吊舱对线路本体、附属设施、通道及电力保护区进行巡检，并记录、整理、上报相关数据的人员，同时协助机长和飞行员保障飞行安全。数据采集要求可见光数据采集4.1.1每基杆塔应按顺序拍摄，保证销钉、螺栓等微小线路元件清晰可见。4.1.2每基杆塔最少拍摄图片要求：直线塔应包含杆塔全塔一张、杆塔电气部分一张、两边地线支架各一张、每相绝缘子串各一张、杆塔基面一张；耐张塔包含杆塔全塔一张、杆塔电气部分一张、两边地线支架各一张、每相绝缘子串各一张、每相跳线串各一张、杆塔基面一张；换位塔可根据跳线情况增加一到二张照片。4.1.3利用照相机采集数据时，应根据杆塔结构尺寸、拍摄位置、巡检重点等合理调整相机焦距。4.1.4利用相机获取可将光缺陷数据时，缺陷部位应位于视场中央且不被其他物体遮挡，同时利用数据记录终端记录缺陷信息。4.1.5利用光电吊舱进行可见光视频录制时应切换宽、窄视场，要求附属设施、通道及电力保护区一起录制并保证画面清晰。红外数据采集4.2.1利用光电吊舱中红外传感器对输电线路全程红外录像，根据巡检需求适时切换远近视场，在显示器上观察是否存在发热缺陷。4.2.2发现红外缺陷或疑点时，巡检员应及时调整红外成像角度，保证其位于视场中央，存储红外热图数据并做好缺陷信息记录。激光点云数据采集4.3.1合理规划作业航迹，激光点云应包含完整的输电线路本体及通道信息。4.3.2激光扫描通道应至少覆盖输电线路杆塔单侧塔高3m范围。4.3.3地面、植被点云平均密度应不小于20points/m2，精细分类时应不小于40points/m2。数据分析要求可见光/红外数据分析5.1.1当天输电线路数据采集结束后，巡检人员及时对缺陷部位、类型、等级、归属进行判定处理。根据当日可见光巡线所发现的缺陷，对可见光巡线图片进行核实，并根据缺陷内容编辑缺陷图片名称。5.1.2利用红外、紫外分析软件分别处理红外、紫外热图，对缺陷进行判定。5.1.3数据采集结束3个工作日内出具最终巡检报告。5.1.4可见光照片主要检查杆塔、电气部分、机械连接和挂点、附属设施等内容。5.1.5可见光视频主要检查通道林木、建筑物及电力保护区及附属设施等内容。5.1.6统计直升机/无人机巡检线路缺陷和隐患数量及巡线飞行时间，将发现的缺陷和隐患归类统计。5.1.7必要时对数据进行二次检查，对于二次检查发现的缺陷可补飞/补测。5.1.8缺陷数据按照《中国南方电网有限责任公司设备缺陷管理办法》，对直升机巡检发现输电线路缺陷进行分类定级并汇报。激光点云数据分析5.2.1坐标应采用WGS-84坐标系或CGCS-2000坐标系。5.2.2激光点云云处理结果应为las格式标准文件。5.2.3利用点云数据处理软件检测输电线路当前工况下安全距离不足的情况。5.2.4根据需要开展输电线路多工况模拟，检测安全距离不足的情况。5.2.5根据需要利用点云数据开展三维建模。5.2.6缺陷数据按照《中国南方电网有限责任公司设备缺陷管理办法》，对直升机巡检发现输电线路缺陷进行分类定级并汇报。数据移交5.2.1直升机/无人机巡检的原始数据、缺陷数据应及时整理，并完成巡检总结报告，包括巡检缺陷记录（见附录A）；5.2.2每条线路巡检结束后，将巡检资料在直升机/无人机电力作业技术支持系统中上传提交，巡检资料移交项目见附录C；5.2.3按季度将巡检资料移交给线路运行管辖单位，移交清单附录D、输电线直升机巡检路缺陷记录单附录E、直升飞机巡检声像资料移交清单附录F等资料5.2.4巡检产生的所有资料必须进行存储备份，巡检资料作业组留存后提交上传至直升机/无人机电力作业技术支持系统，存储在系统数据库中，以便进行资料查询和数据分析；5.2.5直升机巡检作业过程中，紧急缺陷应以电话方式立即通知线路运维单位，紧急、重大缺陷相关影像资料应在1日内发至线路运维单位。5.2.6巡检产生的所有资料必须进行存储备份，以便进行资料查询和数据分析。巡检资料除巡检作业组留存外，至少最近一次巡检数据应在直升机电力作业技术支持系统中存档。5.2.7运维单位根据直升机/无人机电力作业技术支持系统中的疑似缺陷进行核实并消缺。"
    split_text_list = infer.predict(text)
    for text in split_text_list:
        print('---------text----------', text)
            
        
