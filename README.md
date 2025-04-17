# Rasterized Digit Classification on MNIST

## Compile and run CPU native version
```bash
make cpu_native
export OMP_NUM_THREADS=<number of threads>
./model_cpu <Hidden layer 1 size> <Hidden layer 2 size> <Learning rate> <Batch size>
```

## Compile and run CPU BLAS version
```bash
make cpu_blas
export OMP_NUM_THREADS=<number of threads>
./model_blas <Hidden layer 1 size> <Hidden layer 2 size> <Learning rate> <Batch size>
```

## Compile and run GPU native version
```bash
make gpu_native
./<GPU name = a100, v100, rtx> <Hidden layer 1 size> <Hidden layer 2 size> <Learning rate> <Batch size> <Number of blocks> <Number of threads per block>
```

## Compile and run GPU cuBLAS version
```bash
make gpu_cublas
./<GPU name = a100, v100, rtx>_cublas <Hidden layer 1 size> <Hidden layer 2 size> <Learning rate> <Batch size> <Number of blocks> <Number of threads per block>
```

## Sample Output (for alpha = 0.1 and epochs = 50)
```bash
Initialization done, training starts...
Epoch 1 / 50, Avg. training loss = 0.484424,  Avg. validation loss = 0.396270, Test accuracy: 88.919998
Epoch 2 / 50, Avg. training loss = 0.196956,  Avg. validation loss = 0.320438, Test accuracy: 90.760002
Epoch 3 / 50, Avg. training loss = 0.165435,  Avg. validation loss = 0.274980, Test accuracy: 92.029999
Epoch 4 / 50, Avg. training loss = 0.145340,  Avg. validation loss = 0.248819, Test accuracy: 92.769997
Epoch 5 / 50, Avg. training loss = 0.131359,  Avg. validation loss = 0.234762, Test accuracy: 92.949997
.
.
.
Epoch 47 / 50, Avg. training loss = 0.020685,  Avg. validation loss = 0.089256, Test accuracy: 97.479996
Epoch 48 / 50, Avg. training loss = 0.021027,  Avg. validation loss = 0.087876, Test accuracy: 97.599998
Epoch 49 / 50, Avg. training loss = 0.020365,  Avg. validation loss = 0.086338, Test accuracy: 97.680000
Epoch 50 / 50, Avg. training loss = 0.019543,  Avg. validation loss = 0.087212, Test accuracy: 97.549995
Results of model training
Grind rate: 543436
Total training time: 4.600351 seconds
Total inference time: 1.046708 seconds
Learning rate: 0.100000
Batch size: 500
