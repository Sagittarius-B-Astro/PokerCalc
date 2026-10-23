# Build: make        Run tests: make test        Clean: make clean
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

LIB = src/equity.cpp src/features.cpp src/approx.cpp

all: poker validate

poker: src/main.cpp $(LIB) src/*.hpp
	$(CXX) $(CXXFLAGS) src/main.cpp $(LIB) -o poker

validate: tests/validate.cpp src/equity.cpp src/*.hpp
	$(CXX) $(CXXFLAGS) tests/validate.cpp src/equity.cpp -o validate

test: validate
	./validate

clean:
	rm -f poker validate
