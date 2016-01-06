#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CPP = g++
CFLAGS = -lm -pthread -g -O0  -march=native -Wall -funroll-loops -Wno-unused-result -DDEBUG
LIB= -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lOpenCL
all: word2vec 

cbow.o: cbow.cpp
	$(CPP)  -c $< $(LIB) $(CFLAGS)

word2vec: cbow.o word2vec.o
	$(CPP) word2vec.o cbow.o -o $@ $(LIB) $(CFLAGS)
	rm *.o

word2vec.o : word2vec.cpp
	$(CPP) word2vec.cpp -c $< $(LIB) $(CFLAGS)

clean:
	rm -rf word2vec *.o



