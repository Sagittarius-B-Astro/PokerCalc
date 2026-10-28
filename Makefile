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

wasm:
	em++ -O3 -std=c++17 -I src \
	  src/equity.cpp src/features.cpp src/approx.cpp src/wasm_bindings.cpp \
	  --bind \
	  -s MODULARIZE=1 \
	  -s EXPORT_NAME=createPokerModule \
	  -s ALLOW_MEMORY_GROWTH=1 \
	  -s ENVIRONMENT=web,worker \
	  --preload-file data/approx_weights_s0.bin \
	  --preload-file data/approx_weights_s1.bin \
	  --preload-file data/approx_weights_s2.bin \
	  --preload-file data/approx_weights_s3.bin \
	  --preload-file data/approx_weights.bin \
	  -o docs/poker.js