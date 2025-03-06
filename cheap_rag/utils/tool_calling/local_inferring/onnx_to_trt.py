import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path


onnx_dir = Path('/root/nlp/model-params/intfloat/multilingual-e5-large-instruct/onnx')


def build_engine_from_onnx(onnx_file_path, engine_file_path, precision="fp32"):
    """
    将ONNX模型转换为TensorRT引擎
    
    参数:
    onnx_file_path: ONNX模型路径
    engine_file_path: 保存TensorRT引擎的路径
    precision: 精度模式，可选 "fp32"、"fp16" 或 "int8"
    """
    # 创建logger和builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建ONNX解析器
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX文件
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 创建构建配置
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # 设置精度模式
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # 注意：INT8需要校准器，这里简化处理
    
    # 构建并序列化引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎到文件
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"构建完成，TensorRT引擎已保存到 {engine_file_path}")
    return engine

# 使用示例
onnx_model_path = str(onnx_dir.joinpath("model.onnx"))
trt_engine_path = str(onnx_dir.joinpath("model.trt"))
precision = "fp16"  # 可选 "fp32", "fp16", "int8"

engine = build_engine_from_onnx(onnx_model_path, trt_engine_path, precision)