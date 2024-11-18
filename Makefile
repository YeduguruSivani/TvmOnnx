CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -g
INCLUDES = -Iinclude \
          -I/usr/include/opencv4\
		  -I./lib/tvm/3rdparty/dlpack/include \
		  -I./lib/tvm/3rdparty/dmlc-core/include \
		  -I./lib/tvm/src/runtime \
		  -I./lib/tvm/3rdparty/compiler-rt \
		  -I./lib/tvm/include \
          -I./lib/onnxruntime/include

OPENCV_LIBS = -lopencv_videoio \
              -lopencv_video \
              -lopencv_core \
              -lopencv_imgproc \
              -lopencv_imgcodecs \
              -lopencv_highgui \
			  -lpthread \
			  -ltvm_runtime \
			  -lonnxruntime 

LIBS = -L /home/ubuntu6/opencv_20 -L./lib/tvm/build -Wl,-rpath=./lib/tvm/build -L./lib/onnxruntime/lib -Wl,-rpath=./lib/onnxruntime/lib 

TARGET = build
SRC = src
OBJ = obj
DATA = data

all: directories $(TARGET)/people_count

directories:
	mkdir -p $(OBJ) $(TARGET) $(DATA) lib

$(TARGET)/people_count: $(OBJ)/App.o $(OBJ)/main.o $(OBJ)/tvm_detection.o $(OBJ)/onnx_detection.o 
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS) $(LIBS)

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ $(LIBS)

clean:
	rm -f $(OBJ)/* $(TARGET)/*