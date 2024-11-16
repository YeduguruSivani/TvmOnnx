CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
INCLUDES = -Iinclude \
          -I/usr/include/opencv4 \
          -I./lib/onnxruntime/include

OPENCV_LIBS = -lopencv_videoio \
              -lopencv_video \
              -lopencv_core \
              -lopencv_imgproc \
              -lopencv_imgcodecs \
              -lopencv_highgui

ONNX_LIBS = -L./lib/onnxruntime/lib -Wl,-rpath=./lib/onnxruntime/lib -lonnxruntime

TARGET = build
SRC = src
OBJ = obj
DATA = data

SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:$(SRC)/%.cpp=$(OBJ)/%.o)

all: directories $(TARGET)/people_count

directories:
	mkdir -p $(OBJ) $(TARGET) $(DATA) lib

$(TARGET)/people_count: $(OBJ)/App.o $(OBJ)/main.o $(OBJ)/onnx_detection.o # $(OBJ)/tvm_detection.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS) $(ONNX_LIBS)

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ)/* $(TARGET)/*

.PHONY: all directories clean