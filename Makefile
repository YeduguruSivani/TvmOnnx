CXX = g++
CXXFLAGS = -std=c++17 -O2 -w -g
INCLUDES = -Iinclude \
          -I/usr/include/opencv4\
		  -I/home/nvidianx/tvm/3rdparty/dlpack/include \
		  -I/home/nvidianx/tvm/3rdparty/dmlc-core/include \
		  -I/home/nvidianx/tvm/src/runtime \
		  -I/home/nvidianx/tvm/3rdparty/compiler-rt \
		  -I/home/nvidianx/tvm/include \
          	  -I/home/nvidianx/onnxruntime/include/onnxruntime/core/session

OPENCV_LIBS = -lopencv_videoio \
              -lopencv_video \
              -lopencv_core \
              -lopencv_imgproc \
              -lopencv_imgcodecs \
              -lopencv_highgui \
	      -lpthread \
	      -ltvm_runtime \
	      -lonnxruntime 

LIBS = -L /usr/include/opencv4 -L/home/nvidianx/tvm/build -Wl,-rpath=/home/nvidianx/tvm/build -L/home/nvidianx/onnxruntime/build/Linux/Release -Wl,-rpath=/home/nvidianx/onnxruntime/build/Linux/Release 

TARGET = build
SRC = src
OBJ = obj
DATA = data 

all: directories $(TARGET)/people_count

directories:
	@mkdir -p $(OBJ) $(TARGET) $(DATA) lib

$(TARGET)/people_count: $(OBJ)/app.o $(OBJ)/main.o $(OBJ)/i_detector.o $(OBJ)/tvm_detection.o $(OBJ)/onnx_detection.o include/*
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS) $(LIBS)

$(OBJ)/%.o: $(SRC)/%.cpp
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ $(LIBS)

clean:
	@rm -f $(OBJ)/* $(TARGET)/*

refresh: clean all