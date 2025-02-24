# Project 3

# Project 2 - Milestone 1

## Compile and run serial version
```bash
make cpu_model
./model <Hidden layer 1 size> <Hidden layer 2 size> <Learning rate> <Batch size>
```

## Sample Output (for n = 1000 and Nrays = 1 billion)
```bash
Initialization done, training starts...
Epoch 1 / 20 completed, Average train loss = 1.195862, Failed count: 1287, Test accuracy: 87.129997
Epoch 2 / 20 completed, Average train loss = 0.453450, Failed count: 1005, Test accuracy: 89.949997
Epoch 3 / 20 completed, Average train loss = 0.363777, Failed count: 914, Test accuracy: 90.860001
Epoch 4 / 20 completed, Average train loss = 0.325132, Failed count: 841, Test accuracy: 91.589996
Epoch 5 / 20 completed, Average train loss = 0.299900, Failed count: 799, Test accuracy: 92.009995
Epoch 6 / 20 completed, Average train loss = 0.280709, Failed count: 763, Test accuracy: 92.369995
Epoch 7 / 20 completed, Average train loss = 0.264604, Failed count: 693, Test accuracy: 93.070000
Epoch 8 / 20 completed, Average train loss = 0.250740, Failed count: 709, Test accuracy: 92.909996
Epoch 9 / 20 completed, Average train loss = 0.238315, Failed count: 680, Test accuracy: 93.199997
Epoch 10 / 20 completed, Average train loss = 0.226786, Failed count: 621, Test accuracy: 93.790001
Epoch 11 / 20 completed, Average train loss = 0.216631, Failed count: 605, Test accuracy: 93.949997
Epoch 12 / 20 completed, Average train loss = 0.207141, Failed count: 588, Test accuracy: 94.119995
Epoch 13 / 20 completed, Average train loss = 0.198736, Failed count: 556, Test accuracy: 94.439995
Epoch 14 / 20 completed, Average train loss = 0.190763, Failed count: 545, Test accuracy: 94.549995
Epoch 15 / 20 completed, Average train loss = 0.183346, Failed count: 527, Test accuracy: 94.729996
Epoch 16 / 20 completed, Average train loss = 0.176254, Failed count: 525, Test accuracy: 94.750000
Epoch 17 / 20 completed, Average train loss = 0.170156, Failed count: 489, Test accuracy: 95.110001
Epoch 18 / 20 completed, Average train loss = 0.164078, Failed count: 474, Test accuracy: 95.259995
Epoch 19 / 20 completed, Average train loss = 0.158270, Failed count: 462, Test accuracy: 95.379997
Epoch 20 / 20 completed, Average train loss = 0.153197, Failed count: 443, Test accuracy: 95.570000
Results of model training
Grind rate: 14,282
Total training time: 84.019051 seconds
Total inference time: 3.706629 seconds
Learning rate: 0.010000
Batch size: 100
