# TensorFlow Lite

TensorFlow Lite is TensorFlow's lightweight solution for mobile and embedded
devices. It enables low-latency inference of on-device machine learning models
with a small binary size and fast performance supporting hardware acceleration.

See the documentation: https://www.tensorflow.org/mobile/tflite/
Documentation edits can be made here: [tensorflow/docs_src/mobile/tflite](../../docs_src/mobile/tflite)

## Notes
This is modified version of Tensorflow Lite with OpenCL support for Android GPU.

## Some deps
	- OpenCL library (`libOpenCL.so`)
	- OpenCL headers (`cl.h`, `cl_platform.h`)
	- `gnustl_static`

## How to compile (example)
`$ bazel build -c opt --cxxopt='-std=c++0x' --linkopt='-llog' --linkopt='-lOpenCL' --linkopt='-lgnustl_static' //tensorflow/contrib/lite/java:tensorflowlite --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a`

## C++ macros to pass in compilation
	- `FST_IN_BUFF_SIZE` : Buffer size for the first input of conv/matmul
	- `SND_IN_BUFF_SIZE` : Buffer size for the second input of conv/matmul
	- `OUT_BUFF_SIZE` : Buffer size for the output of conv/matmul
	- `MATMUL_WG_HEIGHT` : Work-group height for matmul kernel (width = 4*height)
	- `CONV_WG_HEIGHT` : Work-group height for conv kernel
	- `CONV_WG_WIDTH` : Work-group width for conv kernel

## Android Log Tags
	- `OpenCLDebug` : Contains information about OpenCL kernel errors
	- `ConvRuntime` : Contains information about convolution layers operation (matconv) running time
	- `FCRuntime` : Contains information about fully-connected layers operation (matmul) running time