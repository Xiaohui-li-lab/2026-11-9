CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

TARGET = spin_network

all: $(TARGET)

$(TARGET): spin_network.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) spin_network.cpp

clean:
	rm -f $(TARGET) *.o final_lattice.txt

.PHONY: all clean
