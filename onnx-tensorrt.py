import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# Utility functions
import utils.inference as inference_utils  # TRT/TF inference wrappers

if __name__ == '__main__':

    # Precision command line argument -> TRT Engine datatype
    TRT_PRECISION_TO_DATATYPE = {
        16: trt.DataType.HALF,
        32: trt.DataType.FLOAT
    }
    # datatype: float 32
    trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[32]

    max_batch_size = 1
    # create engine
    dynamic_shapes={"input": ((1, 3, 224, 224), (3, 3, 224, 224), (32, 3, 224, 224))}
    trt_inference_wrapper = inference_utils.TRTInference(
        "./resnet18_dynamic.trt", "./resnet18_dynamic.onnx",
        trt_engine_datatype, max_batch_size, dynamic_shapes
    )
    
    # input 
    dynamic_batch = 6
    input_data = np.ones((dynamic_batch, 3, 224, 224), dtype=np.float32)
    # At runtime you need to set an optimization profile before setting input dimensions. 
    # trt_inference_wrapper.context.active_optimization_profile = 0
    # specifying runtime dimensions
    trt_inference_wrapper.context.set_binding_shape(0, input_data.shape)
    # output
    output_shapes = [(dynamic_batch, 1000)]

    trt_outputs = trt_inference_wrapper.infer(input_data, output_shapes)
    
    print(trt_outputs[0].shape)

    # ----------------------- static batch size -------------------- #
    max_batch_size = 2
    trt_inference_static_wrapper = inference_utils.TRTInference(
        "./resnet18_static.trt", "./resnet18_static.onnx",
        trt_engine_datatype, max_batch_size
    )
    input_data = np.ones((2, 3, 224, 224), dtype=np.float32)
    output_shapes = [(2, 1000)]
    trt_outputs = trt_inference_static_wrapper.infer(input_data, output_shapes)
    
    print(trt_outputs[0].shape)
       