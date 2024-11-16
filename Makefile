# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Paths to dependencies
OPENCV_INCLUDE = -I /usr/include/opencv4
OPENCV_LIB = -L/home/ubuntu6/opencv_20 
TVM_INCLUDE = -I /home/ubuntu6/tvm/3rdparty/dlpack/include \
			-I /home/ubuntu6/tvm/3rdparty/dmlc-core/include \
			-I /home/ubuntu6/tvm/src/runtime \
			-I /home/ubuntu6/tvm/3rdparty/compiler-rt \
			-I /home/ubuntu6/tvm/include 
TVM_LIB = -L/home/ubuntu6/tvm/build -Wl,-rpath=/home/ubuntu6/tvm/build
ONNX_INCLUDE = 
ONNX_LIB = 

# Output executable
TARGET = app

# Source files
SRCS = main.cpp SafeQueue.cpp Detector.cpp App.cpp TVMDetector.cpp ONNXDetector.cpp
OBJS = $(SRCS:.cpp=.o)

# Libraries
LIBS = -lopencv_videoio \
    -lopencv_video \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -ltvm_runtime \
    -lpthread 
# Include directories
INCLUDES = -I$(OPENCV_INCLUDE) -I$(TVM_INCLUDE) -I$(ONNX_INCLUDE)

# Library directories
LIBDIRS = -L$(OPENCV_LIB) -L$(TVM_LIB) -L$(ONNX_LIB)

# Build rules
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LIBDIRS) $(LIBS)

# Object file rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
