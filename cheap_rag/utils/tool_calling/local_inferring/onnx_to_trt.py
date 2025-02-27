import tensorrt as trt
import onnx


# 加载 ONNX 模型
onnx_model = onnx.load('/root/model-params/multilingual-e5-large-instruct/onnx/model.onnx')

# 创建 TensorRT 的 Builder 和 Network
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(trt.NetworkDefinitionCreationFlags.EXPLICIT_BATCH)
parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

# 解析 ONNX 模型
with open('/root/model-params/multilingual-e5-large-instruct/onnx/model.onnx', 'rb') as f:
    parser.parse(f.read())

# 创建 TensorRT 引擎
engine = builder.build_cuda_engine(network)

# 将引擎保存到文件
with open('/root/model-params/multilingual-e5-large-instruct/onnx/model_trt.engine', 'wb') as f:
    f.write(engine.serialize())
