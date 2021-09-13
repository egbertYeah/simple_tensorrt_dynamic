# simple_tensorrt_dynamic
a simple example to learn tensorrt with dynamic shapes

# 依赖环境
- Ubuntu 18.04
- pytorch>=1.4
- onnx 1.10
- tensorrt 7.2.2

# 实现功能
以resnet18为例，实现了dynamic和static下engine的生成，并提供一个可用重复使用的框架，便于常见torch->onnx->tensorrt方案的实现。

# 如何运行？
1. 生成onnx文件
```
  python torch-onnx.py
```
2. 生成engine文件并推理
```
  python onnx-tensorrt.py
```
