CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Iinclude

#OPENCV
OPENCV_INCLUDE = -I /usr/include/opencv4
OPENCV_LIB = -L/home/ubuntu6/opencv_20 
#TVM
TVM_INCLUDE = -I /home/ubuntu6/tvm/3rdparty/dlpack/include \
			-I /home/ubuntu6/tvm/3rdparty/dmlc-core/include \
			-I /home/ubuntu6/tvm/src/runtime \
			-I /home/ubuntu6/tvm/3rdparty/compiler-rt \
			-I /home/ubuntu6/tvm/include 
TVM_LIB = -L/home/ubuntu6/tvm/build -Wl,-rpath=/home/ubuntu6/tvm/build
#ONNX
ONNX_INCLUDE = -I./lib/onnxruntime/include
ONNX_LIB = -L./lib/onnxruntime/lib -Wl,-rpath=./lib/onnxruntime/lib 

LIB_FLAGS = -lopencv_videoio \
    -lopencv_video \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -ltvm_runtime \
    -lpthread  \
	-lonnxruntime

TARGET = build
SRC = src
OBJ = obj
DATA = data

INCLUDES = $(OPENCV_INCLUDE) $(ONNX_INCLUDE) $(TVM_INCLUDE) 
LIBS = $(OPENCV_LIB) $(ONNX_LIB) $(TVM_LIB) 

all: $(TARGET)/people_count

directories:
	mkdir -p $(OBJ) $(TARGET) $(DATA)

$(TARGET)/people_count: $(OBJ)/main.o $(OBJ)/onnx_detection.o $(OBJ)/tvm_detection.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $^ -o $@ $(LIB_FLAGS)

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(OBJ)/* $(TARGET)/*