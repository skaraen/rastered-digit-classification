#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "mnist.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define EPOCHS 50
#define TILE_SIZE 32 

#define min(a,b) ((a) < (b) ? (a) : (b))

__host__ float xavier(int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    return ((float)rand() / RAND_MAX) * 2 * limit - limit;
}

// Inspired from Fischer Yates shuffle algorithm
__host__ void shuffle_data(float* images, int* labels, int num_images, int img_dim) {
    for (int i = num_images - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        for (int k = 0; k < img_dim; k++) {
            float temp = images[i * img_dim + k];
            images[i * img_dim + k] = images[j * img_dim + k];
            images[j * img_dim + k] = temp;
        }

        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}

__global__ void matrix_add_kernel(float* res, float* a, float* b, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        res[idx] = a[idx] + b[idx];
    }
}

__global__ void matrix_difference_kernel(float* res, float* a, float* b, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        res[idx] = a[idx] - b[idx];
    }
}

__global__ void matrix_multiply_kernel(float* res, float* a, float* b, int x, int y, int z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / z;
    int k = idx % z;
    
    if (i < x && k < z) {
        float sum = 0.0f;
        for (int j = 0; j < y; j++) {
            sum += a[i * y + j] * b[j * z + k];
        }
        res[i * z + k] = sum;
    }
}

__global__ void matrix_multiply_transpose1_kernel(float* res, float* a, float* b, int x, int y, int z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / z;
    int k = idx % z;
    
    if (i < y && k < z) {
        float sum = 0.0f;
        for (int j = 0; j < x; j++) {
            sum += a[j * y + i] * b[j * z + k];
        }
        res[i * z + k] = sum;
    }
}

__global__ void matrix_multiply_transpose2_kernel(float* res, float* a, float* b, int x, int y, int z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / z;
    int k = idx % z;
    
    if (i < x && k < z) {
        float sum = 0.0f;
        for (int j = 0; j < y; j++) {
            sum += a[i * y + j] * b[k * y + j];
        }
        res[i * z + k] = sum;
    }
}

__global__ void matrix_copy_kernel(float* res, float* a, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        res[idx] = a[idx];
    }
}

__global__ void relu_kernel(float* res, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        res[idx] = fmaxf(0, res[idx]);
    }
}

__global__ void softmax_kernel(float* res, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bId = blockIdx.x;
    int tId = threadIdx.x;

    if (bId < y) {
        __shared__ float sum;

        if (tId == 0) {
            sum = 0.0f;
        }
        __syncthreads();

        if (tId < x) {
            float val = expf(res[tId * y + bId]);
            atomicAdd(&sum, val);
            __syncthreads();

            res[tId * y + bId] = val / sum;
        }
    }
}

__global__ void update_weights_kernel(float *w, float *del_w, int x, int y, float alpha, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        w[idx] -= (alpha * del_w[idx]) / m;
        del_w[idx] = 0.0f;
    }
}

__global__ void update_biases_kernel(float *b, float *del_b, int x, int y, float alpha, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        b[idx] -= (alpha * del_b[idx]) / m;
        del_b[idx] = 0.0f;
    }
}

__global__ void reset_kernel(float *a, int x, int y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < x * y) {
        a[idx] = 0.0f;
    }
}

__global__ void result_kernel(float *y, int *train_label, int k, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m) {
        int id = k * m + idx;
        y[train_label[id] * m + idx] = 1.0f;
    }
}

__global__ void sgd_relu_kernel(float *error, float *h, float *y, float *del_b, int n_h, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_h * m) {
        error[idx] = (h[idx] > 0) ? y[idx] : 0;
        del_b[idx] = error[idx];
    }
}

__global__ void loss_kernel(float* d_C, float *d_out, int *d_train_label, int k, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (blockIdx.x == 0) {
        __shared__ float C;

        if (threadIdx.x == 0) {
            C = 0.0f;
        }
        __syncthreads();

        if (threadIdx.x < m) {
            int id = k * m + threadIdx.x;
            float val = -log(d_out[d_train_label[id] * m + threadIdx.x]);
            atomicAdd(&C, val);
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(d_C, C);
        }
    }   
}

__host__ void weighted_product_relu(float* d_res, float* d_w, float* d_a, float *d_b, int x, int y, int z, int nblocks, int ntpb, cublasHandle_t handle, float* alp, float* bet) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, z, x, y, alp, d_a, z, d_w, y, bet, d_res, z);
    matrix_add_kernel<<<nblocks, ntpb>>>(d_res, d_res, d_b, x, z);
    relu_kernel<<<nblocks, ntpb>>>(d_res, x, z);
}

__host__ void weighted_product_softmax(float* d_res, float* d_w, float* d_a, float *d_b, int x, int y, int z, int nblocks, int ntpb, cublasHandle_t handle, float* alp, float* bet) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, z, x, y, alp, d_a, z, d_w, y, bet, d_res, z);
    matrix_add_kernel<<<nblocks, ntpb>>>(d_res, d_res, d_b, x, z);
    softmax_kernel<<<nblocks, ntpb>>>(d_res, x, z);
}

__host__ float* init_1D(int m, int n) {
    float *mat = (float*) malloc(m * n * sizeof(float));

    return mat;
}

__host__ float* init_xavier(int m, int n) {
    float *mat = init_1D(m, n);
    for (int i = 0; i < m * n; i++) {
        mat[i] = xavier(m, n);
    }

    return mat;
}

__host__ float* init_zero(int m, int n) {
    float *mat = init_1D(m, n);
    for (int i = 0; i < m * n; i++) {
        mat[i] = 0.0f;
    }

    return mat;
}

int main(int argc, char** argv) {
    cudaEvent_t start_train, end_train, start_test, end_test;

    cudaEventCreate(&start_train);
    cudaEventCreate(&end_train);
    cudaEventCreate(&start_test);
    cudaEventCreate(&end_test);

    int n_h1 = atoi(argv[1]);
    int n_h2 = atoi(argv[2]);
    float alpha = atof(argv[3]);
    int m = atoi(argv[4]);
    int nblocks = atoi(argv[5]);
    int ntpb = atoi(argv[6]);

    float alp = 1.0;
    float bet = 0.0;

    load_mnist();

    // Initialize network layers
    float *in = init_1D(INPUT_SIZE, m);
    float *h1 = init_1D(n_h1, m);
    float *h2 = init_1D(n_h2, m);
    float *out = init_1D(OUTPUT_SIZE, m);

    // Initialize weights
    float *w1 = init_xavier(n_h1, INPUT_SIZE);
    float *w2 = init_xavier(n_h2, n_h1);
    float *w3 = init_xavier(OUTPUT_SIZE, n_h2);

    // Initialize biases
    float *b1 = init_zero(n_h1, m);
    float *b2 = init_zero(n_h2, m);
    float *b3 = init_zero(OUTPUT_SIZE, m);

    // Initialize delta weights and biases
    float *del_w1 = init_zero(n_h1, INPUT_SIZE);
    float *del_w2 = init_zero(n_h2, n_h1);
    float *del_w3 = init_zero(OUTPUT_SIZE, n_h2);

    float *del_b1 = init_zero(n_h1, m);
    float *del_b2 = init_zero(n_h2, m);
    float *del_b3 = init_zero(OUTPUT_SIZE, m);

    float *error_out = init_zero(OUTPUT_SIZE, m);
    float *error_h2 = init_zero(n_h2, m);
    float *error_h1 = init_zero(n_h1, m);

    float *y = init_zero(OUTPUT_SIZE, m);
    float *y_h2 = init_zero(n_h2, m);
    float *y_h1 = init_zero(n_h1, m);

    // Initialize network layers
    float *d_in, *d_h1, *d_h2, *d_out, *d_w1, *d_w2, *d_w3, *d_b1, *d_b2, *d_b3, *d_del_w1, *d_del_w2, *d_del_w3, *d_del_b1, *d_del_b2, *d_del_b3, *d_error_out, *d_error_h1, *d_error_h2, *d_y, *d_y_h1, *d_y_h2, *d_C_train;

    cudaMalloc((void**)&d_C_train, sizeof(float));

    int *d_train_label, *d_valid_label, *d_test_label;
    cudaMalloc((void**)&d_train_label, NUM_TRAIN * sizeof(int));
    cudaMalloc((void**)&d_valid_label, NUM_VALID * sizeof(int));
    cudaMemcpy(d_valid_label, valid_label, NUM_VALID * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_test_label, NUM_TEST * sizeof(int));
    cudaMemcpy(d_test_label, test_label, NUM_TEST * sizeof(int), cudaMemcpyHostToDevice);

     // Initialize network layers
    cudaMalloc((void**)&d_in, INPUT_SIZE * m * sizeof(float));
    cudaMalloc((void**)&d_h1, n_h1 * m * sizeof(float));
    cudaMalloc((void**)&d_h2, n_h2 * m * sizeof(float));
    cudaMalloc((void**)&d_out, OUTPUT_SIZE * m * sizeof(float));

    // Initialize weights
    cudaMalloc((void**)&d_w1, n_h1 * INPUT_SIZE * sizeof(float));
    cudaMemcpy(d_w1, w1, n_h1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_w2, n_h2 * n_h1 * sizeof(float));
    cudaMemcpy(d_w2, w2, n_h2 * n_h1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_w3, OUTPUT_SIZE * n_h2 * sizeof(float));
    cudaMemcpy(d_w3, w3, OUTPUT_SIZE * n_h2 * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize biases
    cudaMalloc((void**)&d_b1, n_h1 * m * sizeof(float));
    cudaMemcpy(d_b1, b1, n_h1 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_b2, n_h2 * m * sizeof(float));
    cudaMemcpy(d_b2, b2, n_h2 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_b3, OUTPUT_SIZE * m * sizeof(float));
    cudaMemcpy(d_b3, b3, OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize delta weights and biases
    cudaMalloc((void**)&d_del_w1, n_h1 * INPUT_SIZE * sizeof(float));
    cudaMemcpy(d_del_w1, del_w1, n_h1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_del_w2, n_h2 * n_h1 * sizeof(float));
    cudaMemcpy(d_del_w2, del_w2, n_h2 * n_h1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_del_w3, OUTPUT_SIZE * n_h2 * sizeof(float));
    cudaMemcpy(d_del_w3, del_w3, OUTPUT_SIZE * n_h2 * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_del_b1, n_h1 * m * sizeof(float));
    cudaMemcpy(d_del_b1, del_b1,  n_h1 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_del_b2, n_h2 * m * sizeof(float));
    cudaMemcpy(d_del_b2, del_b2, n_h2 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_del_b3, OUTPUT_SIZE * m * sizeof(float));
    cudaMemcpy(d_del_b3, del_b3, OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_error_out, OUTPUT_SIZE * m * sizeof(float));
    cudaMemcpy(d_error_out, error_out,  OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_error_h2, n_h2 * m * sizeof(float));
    cudaMemcpy(d_error_h2, error_h2, n_h2 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_error_h1, n_h1 * m * sizeof(float));
    cudaMemcpy(d_error_h1, error_h1, n_h1 * m * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_y, OUTPUT_SIZE * m * sizeof(float));
    cudaMemcpy(d_y, y,  OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_y_h2, n_h2 * m * sizeof(float));
    cudaMemcpy(d_y_h2, y_h2, n_h2 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_y_h1, n_h1 * m * sizeof(float));
    cudaMemcpy(d_y_h1, y_h1, n_h1 * m * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("Initialization done, training starts...\n");

    float total_train_time = 0.0f;
    float total_test_time = 0.0f;
    cudaDeviceSynchronize();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float *C_train = (float*) calloc(1, sizeof(float));
        cudaMemcpy(d_C_train, C_train, sizeof(float), cudaMemcpyHostToDevice);
        float C_valid = 0.0f;
        shuffle_data(&train_image[0][0], train_label, NUM_TRAIN, INPUT_SIZE);
        cudaMemcpy(d_train_label, train_label, NUM_TRAIN * sizeof(int), cudaMemcpyHostToDevice);

        // Train network
        cudaEventRecord(start_train, 0);
        for (int k = 0; k < NUM_TRAIN / m; k++) {
            float C = 0.0f;

            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i * m + x] = train_image[(k * m) + x][i];
                }
            }
            cudaMemcpy(d_in, in,  INPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            // Feed-forward
            weighted_product_relu(d_h1, d_w1, d_in, d_b1, n_h1, INPUT_SIZE, m, nblocks, ntpb, handle, &alp, &bet);
            weighted_product_relu(d_h2, d_w2, d_h1, d_b2, n_h2, n_h1, m, nblocks, ntpb, handle, &alp, &bet);
            weighted_product_softmax(d_out, d_w3, d_h2, d_b3, OUTPUT_SIZE, n_h2, m, nblocks, ntpb, handle, &alp, &bet);

            loss_kernel<<<nblocks,ntpb>>>(d_C_train, d_out, d_train_label, k, m);

            // SGD calculation
            // Output layer
            result_kernel<<<nblocks,ntpb>>>(d_y, d_train_label, k, m);
            matrix_difference_kernel<<<nblocks,ntpb>>>(d_error_out, d_out, d_y, OUTPUT_SIZE, m);
            matrix_copy_kernel<<<nblocks,ntpb>>>(d_del_b3, d_error_out, OUTPUT_SIZE, m);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_h2, OUTPUT_SIZE, m, &alp, d_h2, m, d_error_out, m, &bet, d_del_w3, n_h2);

            // Hidden layer 2
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n_h2, OUTPUT_SIZE, &alp, d_error_out, m, d_w3, n_h2, &bet, d_y_h2, m);
            sgd_relu_kernel<<<nblocks, ntpb>>>(d_error_h2, d_h2, d_y_h2, d_del_b2, n_h2, m);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_h1, n_h2, m, &alp, d_h1, m, d_error_h2, m, &bet, d_del_w2, n_h1);

            // Hidden layer 1
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n_h1, n_h2, &alp, d_error_h2, m, d_w2, n_h1, &bet, d_y_h1, m);
            sgd_relu_kernel<<<nblocks, ntpb>>>(d_error_h1, d_h1, d_y_h1, d_del_b1, n_h1, m);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, INPUT_SIZE, n_h1, m, &alp, d_in, m, d_error_h1, m, &bet, d_del_w1, INPUT_SIZE);

            update_weights_kernel<<<nblocks, ntpb>>>(d_w1, d_del_w1, n_h1, INPUT_SIZE, alpha, m);
            update_weights_kernel<<<nblocks, ntpb>>>(d_w2, d_del_w2, n_h2, n_h1, alpha, m);
            update_weights_kernel<<<nblocks, ntpb>>>(d_w3, d_del_w3, OUTPUT_SIZE, n_h2, alpha, m);

            update_biases_kernel<<<nblocks, ntpb>>>(d_b1, d_del_b1, n_h1, m, alpha, m);
            update_biases_kernel<<<nblocks, ntpb>>>(d_b2, d_del_b2, n_h2, m, alpha, m);
            update_biases_kernel<<<nblocks, ntpb>>>(d_b3, d_del_b3, OUTPUT_SIZE, m, alpha, m);

            reset_kernel<<<nblocks, ntpb>>>(d_y, OUTPUT_SIZE, m);
        }
        cudaMemcpy(C_train, d_C_train, sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(end_train, 0);
        cudaEventSynchronize(end_train);

        float train_time = 0.0f;
        cudaEventElapsedTime(&train_time, start_train, end_train);

        total_train_time += train_time;

        // Validation
        for (int k = 0; k < NUM_VALID / m; k++) {
            float C = 0.0f;
            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) { 
                    in[i * m + x] = valid_image[k * m + x][i];
                }
            }
            cudaMemcpy(d_in, in,  INPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            weighted_product_relu(d_h1, d_w1, d_in, d_b1, n_h1, INPUT_SIZE, m, nblocks, ntpb, handle, &alp, &bet);
            weighted_product_relu(d_h2, d_w2, d_h1, d_b2, n_h2, n_h1, m, nblocks, ntpb, handle, &alp, &bet);
            weighted_product_softmax(d_out, d_w3, d_h2, d_b3, OUTPUT_SIZE, n_h2, m, nblocks, ntpb, handle, &alp, &bet);

            cudaMemcpy(out, d_out, OUTPUT_SIZE * m * sizeof(float), cudaMemcpyDeviceToHost);

            for (int x = 0; x < m; x++) {
                int id = k * m + x;
                C += -log(out[valid_label[id] * m + x]);
            }
            C_valid += C;
        }

        // Testing
        cudaEventRecord(start_test, 0);
        int fail_ct = 0;
        for (int k = 0; k < NUM_TEST / m; k++) {
            cudaDeviceSynchronize();
            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i * m + x] = test_image[(k * m) + x][i];
                }
            }
            cudaMemcpy(d_in, in,  INPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            weighted_product_relu(d_h1, d_w1, d_in, d_b1, n_h1, INPUT_SIZE, m, nblocks, ntpb, handle, &alp, &bet);
            weighted_product_relu(d_h2, d_w2, d_h1, d_b2, n_h2, n_h1, m, nblocks, ntpb, handle, &alp, &bet);
            weighted_product_softmax(d_out, d_w3, d_h2, d_b3, OUTPUT_SIZE, n_h2, m, nblocks, ntpb, handle, &alp, &bet);

            cudaMemcpy(out, d_out, OUTPUT_SIZE * m * sizeof(float), cudaMemcpyDeviceToHost);

            for (int x = 0; x < m; x++) {
                int prediction;
                float max_act = 0.0f;
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    int idx = i * m + x;
                    if (max_act < out[idx]) {
                        max_act = out[idx];
                        prediction = i;
                    }
                }

                if (prediction != test_label[k * m + x]) {
                    fail_ct++;
                }
            }
        }
        cudaEventRecord(end_test, 0);
        cudaEventSynchronize(end_test);

        float test_time;
        cudaEventElapsedTime(&test_time, start_test, end_test);

        total_test_time += test_time;

        printf("Epoch %d / %d, Avg. training loss = %f,  Avg. validation loss = %f, Test accuracy: %f\n", epoch + 1, EPOCHS, *C_train / NUM_TRAIN, C_valid / NUM_VALID, ((float) (NUM_TEST - fail_ct) / NUM_TEST) * 100.0f);
    }

    free(in);
    free(w1);
    free(del_w1);
    free(h1);
    free(b1);
    free(del_b1);
    free(error_h1);
    free(y_h1);
    free(w2);
    free(del_w2);
    free(h2);
    free(b2);
    free(del_b2);
    free(error_h2);
    free(y_h2);
    free(w3);
    free(del_w3);
    free(out);
    free(b3);
    free(del_b3);
    free(error_out);
    free(y);

    cudaFree(d_in);
    cudaFree(d_w1);
    cudaFree(d_del_w1);
    cudaFree(d_h1);
    cudaFree(d_b1);
    cudaFree(d_del_b1);
    cudaFree(d_error_h1);
    cudaFree(d_y_h1);
    cudaFree(d_w2);
    cudaFree(d_del_w2);
    cudaFree(d_h2);
    cudaFree(d_b2);
    cudaFree(d_del_b2);
    cudaFree(d_error_h2);
    cudaFree(d_y_h2);
    cudaFree(d_w3);
    cudaFree(d_del_w3);
    cudaFree(d_out);
    cudaFree(d_b3);
    cudaFree(d_del_b3);
    cudaFree(d_error_out);
    cudaFree(d_y);
    cudaFree(d_train_label);
    cudaFree(d_valid_label);
    cudaFree(d_test_label);
    cublasDestroy(handle);

    printf("Results of model training\n");
    printf("Grind rate: %d\n", (int)((float)(NUM_TRAIN * EPOCHS) / total_train_time * 1000));
    printf("Total training time: %f seconds\n", total_train_time / 1000);
    printf("Total inference time: %f seconds\n", total_test_time / 1000);
    printf("Learning rate: %f\n", alpha);
    printf("Batch size: %d\n", m);
}
