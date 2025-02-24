CC = gcc
CFLAGS = -O3 -march=native -flto -funroll-loops -ffast-math -lm

all: cpu_model

cpu_model: model.c
	$(CC) $(CFLAGS) model.c -o model

clean:
	rm -f model
