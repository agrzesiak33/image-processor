all: med_filt

med_filt: main.o image_matrix.o
	g++ -fopenmp main.o image_matrix.o -o med_filt

main.o: main.cpp image_matrix.hpp
	g++ -fopenmp -c main.cpp

image_matrix.o: image_matrix.cpp image_matrix.hpp
	g++ -c image_matrix.cpp

clean:
	rm -rf *.o med_filt
