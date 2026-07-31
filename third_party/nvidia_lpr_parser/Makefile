CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -shared -fPIC
CXXFLAGS += -I/opt/nvidia/deepstream/deepstream/sources/includes
CXXFLAGS += -I/usr/local/cuda/targets/x86_64-linux/include
TARGET := libnvdsinfer_custom_impl_lpr.so

all: $(TARGET)

$(TARGET): nvinfer_custom_lpr_parser.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	rm -f $(TARGET)
