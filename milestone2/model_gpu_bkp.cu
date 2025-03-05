#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mnist_gpu.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define EPOCHS 5
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < x && j < y) {
        int index = i * y + j;
        res[index] = a[index] + b[index];
    }
}

__global__ void matrix_difference_kernel(float* res, float* a, float* b, int x, int y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < x && j < y) {
        int index = i * y + j;
        res[index] = a[index] - b[index];
    }
}

__host__ void matrix_difference(float* res, float* a, float* b, int x, int y) {
    for (int i = 0; i < x * y; i++) {
        res[i] = a[i] - b[i];
    }
}

__global__ void matrix_multiply_kernel(float* res, float* a, float* b, int x, int y, int z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < x && k < z) {
        float sum = 0.0f;
        for (int j = 0; j < y; j++) {
            sum += a[i * y + j] * b[j * z + k];
        }
        res[i * z + k] = sum;
    }
}

__global__ void matrix_multiply_transpose1_kernel(float* res, float* a, float* b, int x, int y, int z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < y && k < z) {
        float sum = 0.0f;
        for (int j = 0; j < x; j++) {
            sum += a[j * y + i] * b[j * z + k];
        }
        res[i * z + k] = sum;
    }
}

__global__ void matrix_multiply_transpose2_kernel(float* res, float* a, float* b, int x, int y, int z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < x && k < z) {
        float sum = 0.0f;
        for (int j = 0; j < y; j++) {
            sum += a[i * y + j] * b[k * y + j];
        }
        res[i * z + k] = sum;
    }
}

__host__ void matrix_copy(float* res, float* a, int x, int y) {
    for (int i = 0; i < x * y; i++) {
        res[i] = a[i];
    }
}

__host__ void weighted_product_relu(float* h_res, float* d_res, float* d_w, float* d_a, float *d_b, int x, int y, int z, int nblocks, int ntpb) {
    matrix_multiply_kernel<<<nblocks, ntpb>>>(d_res, d_w, d_a, x, y, z);
    cudaDeviceSynchronize();
    matrix_add_kernel<<<nblocks, ntpb>>>(d_res, d_res, d_b, x, z);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res, x * z * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < x * z; i++) {
        h_res[i] = fmaxf(0, h_res[i]);
    }
}

__host__ void weighted_product_sigmoid(float* h_res, float* d_res, float* d_w, float* d_a, float *d_b, int x, int y, int z, int nblocks, int ntpb) {
    matrix_multiply_kernel<<<nblocks, ntpb>>>(d_res, d_w, d_a, x, y, z);
    cudaDeviceSynchronize();
    matrix_add_kernel<<<nblocks, ntpb>>>(d_res, d_res, d_b, x, z);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res, x * z * sizeof(float), cudaMemcpyDeviceToHost);

    float sum[z];
    for (int k = 0; k < z; k++) {
        sum[k] = 0.0f;
        for (int i = 0; i < x; i++) {
            sum[k] += expf(h_res[i * z + k]);
        }
    }

    for (int i = 0; i < x; i++) {
        for (int k = 0; k < z; k++) {
            h_res[i * z + k] = expf(h_res[i * z + k]) / sum[k];
        }
    }
}

__host__ float* init_2D(int m, int n) {
    float *mat = (float*) malloc(m * n * sizeof(float));

    return mat;
}

__host__ float* init_xavier(int m, int n) {
    float *mat = init_2D(m, n);
    for (int i = 0; i < m * n; i++) {
        mat[i] = xavier(m, n);
    }

    return mat;
}

__host__ float* init_zero(int m, int n) {
    float *mat = init_2D(m, n);
    for (int i = 0; i < m * n; i++) {
        mat[i] = 0.0f;
    }

    return mat;
}

__host__ void update_weights(float *w, float *del_w, int x, int y, float alpha, int m) {
    for (int i = 0; i < x * y; i++) {
        w[i] -= (alpha * del_w[i]) / m;
        del_w[i] = 0.0f;
    }
}

__host__ void update_biases(float *b, float *del_b, int x, int y, float alpha, int m) {
    for (int i = 0; i < x * y; i++) {
        b[i] -= (alpha * del_b[i]) / m;
        del_b[i] = 0.0f;
    }
}

__host__ void reset(float *a, int x, int y) {
    for (int i = 0; i < x * y; i++) {
        a[i] = 0.0f;
    }
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

    // load_mnist();

    mnist_data *train_data;
    unsigned int train_cnt;
    int ret;

    if (ret = mnist_load("./data/train-images-idx3-ubyte", "/data/train-labels-idx1-ubyte", &train_data, &train_cnt)) {
        printf("An error occured: %d\n", ret);
    } else {
        printf("Train image count: %d\n", cnt);
    }

    mnist_data *test_data;
    unsigned int test_cnt;

    if (ret = mnist_load("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", &test_data, &test_cnt)) {
        printf("An error occured: %d\n", ret);
    } else {
        printf("Test image count: %d\n", cnt);
    }

    // Initialize network layers
    float *in = init_2D(INPUT_SIZE, m);
    float *h1 = init_2D(n_h1, m);
    float *h2 = init_2D(n_h2, m);
    float *out = init_2D(OUTPUT_SIZE, m);

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
    float *d_in, *d_h1, *d_h2, *d_out, *d_w1, *d_w2, *d_w3, *d_b1, *d_b2, *d_b3, *d_del_w1, *d_del_w2, *d_del_w3, *d_del_b1, *d_del_b2, *d_del_b3, *d_error_out, *d_error_h1, *d_error_h2, *d_y, *d_y_h1, *d_y_h2;

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

    printf("Initialization done, training starts...\n");

    float total_train_time, total_test_time = 0.0f;
    double start_time, end_time;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float C_train = 0.0f;
        float C_test = 0.0f;
        // shuffle_data(&train_image[0][0], train_label, NUM_TRAIN, INPUT_SIZE);

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
            weighted_product_relu(h1, d_h1, d_w1, d_in, d_b1, n_h1, INPUT_SIZE, m, nblocks, ntpb);
            weighted_product_relu(h2, d_h2, d_w2, d_h1, d_b2, n_h2, n_h1, m, nblocks, ntpb);
            weighted_product_sigmoid(out, d_out, d_w3, d_h2, d_b3, OUTPUT_SIZE, n_h2, m, nblocks, ntpb);

            for (int x = 0; x < m; x++) {
                int id = k * m + x;
                C += -log(out[train_label[id] * m + x]);
            }

            // SGD calculation
            // Output layer
            for (int x = 0; x < m; x++) {
                int id = k * m + x;
                y[train_label[id] * m + x] = 1.0f;
            }

            matrix_difference(error_out, out, y, OUTPUT_SIZE, m);
            matrix_copy(del_b3, error_out, OUTPUT_SIZE, m);
            cudaMemcpy(d_error_out, error_out,  OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            matrix_multiply_transpose2_kernel<<<nblocks, ntpb>>>(d_del_w3, d_error_out, h2, OUTPUT_SIZE, m, n_h2);
            cudaDeviceSynchronize();
            cudaMemcpy(del_w3, d_del_w3, OUTPUT_SIZE * n_h2 * sizeof(float), cudaMemcpyDeviceToHost);

            // Hidden layer 2
            matrix_multiply_transpose1_kernel<<<nblocks, ntpb>>>(d_y_h2, d_w3, d_error_out, OUTPUT_SIZE, n_h2, m);
            cudaDeviceSynchronize();
            cudaMemcpy(y_h2, d_y_h2, n_h2 * m * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < n_h2 * m; i++) {
                error_h2[i] = (h2[i] > 0) ? y_h2[i] : 0;
                del_b2[i] = error_h2[i];
            }
            cudaMemcpy(d_error_h2, error_h2, n_h2 * m * sizeof(float), cudaMemcpyHostToDevice);

            matrix_multiply_transpose2_kernel<<<nblocks, ntpb>>>(d_del_w2, d_error_h2, d_h1, n_h2, m, n_h1);
            cudaDeviceSynchronize();
            cudaMemcpy(del_w2, d_del_w2, n_h2 * n_h1 * sizeof(float), cudaMemcpyDeviceToHost);

            // Hidden layer 1
            matrix_multiply_transpose1_kernel<<<nblocks, ntpb>>>(d_y_h1, d_w2, d_error_h2, n_h2, n_h1, m);
            for (int i = 0; i < n_h1 * m; i++) {
                error_h1[i] = (h1[i] > 0) ? y_h1[i] : 0;
                del_b1[i] = error_h1[i];
            }
            cudaMemcpy(d_error_h1, error_h1, n_h1 * m * sizeof(float), cudaMemcpyHostToDevice);

            matrix_multiply_transpose2_kernel<<<nblocks, ntpb>>>(d_del_w1, d_error_h1, d_in, n_h1, m, INPUT_SIZE);
            cudaDeviceSynchronize();
            cudaMemcpy(del_w1, d_del_w1, n_h1 * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            update_weights(w1, del_w1, n_h1, INPUT_SIZE, alpha, m);
            update_weights(w2, del_w2, n_h2, n_h1, alpha, m);
            update_weights(w3, del_w3, OUTPUT_SIZE, n_h2, alpha, m);

            update_biases(b1, del_b1, n_h1, m, alpha, m);
            update_biases(b2, del_b2, n_h2, m, alpha, m);
            update_biases(b3, del_b3, OUTPUT_SIZE, m, alpha, m);
    
            cudaMemcpy(d_w1, w1, n_h1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_w2, w2, n_h2 * n_h1 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_w3, w3, OUTPUT_SIZE * n_h2 * sizeof(float), cudaMemcpyHostToDevice);

            cudaMemcpy(d_b1, b1, n_h1 * m * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b2, b2, n_h2 * m * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b3, b3, OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            C_train += C;
            reset(y, OUTPUT_SIZE, m);
            cudaMemcpy(d_y, y,  OUTPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
                exit(1);
            }
        }
        cudaEventRecord(end_train, 0);
        cudaEventSynchronize(end_train);

        float train_time;
        cudaEventElapsedTime(&train_time, start_train, end_train);

        total_train_time += train_time;

        // Testing
        cudaEventRecord(start_test, 0);
        int fail_ct = 0;
        for (int k = 0; k < NUM_TEST / m; k++) {
            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i * m + x] = test_image[(k * m) + x][i];
                }
            }
            cudaMemcpy(d_in, in,  INPUT_SIZE * m * sizeof(float), cudaMemcpyHostToDevice);

            weighted_product_relu(h1, d_h1, d_w1, d_in, d_b1, n_h1, INPUT_SIZE, m, nblocks, ntpb);
            weighted_product_relu(h2, d_h2, d_w2, d_h1, d_b2, n_h2, n_h1, m, nblocks, ntpb);
            weighted_product_sigmoid(out, d_out, d_w3, d_h2, d_b3, OUTPUT_SIZE, n_h2, m, nblocks, ntpb);

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

        printf("Epoch %d / %d completed, Average train loss = %f, Failed count: %d, Test accuracy: %f\n", epoch + 1, EPOCHS, C_train / NUM_TRAIN, fail_ct, ((float) (NUM_TEST - fail_ct) / NUM_TEST) * 100.0f);
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

    printf("Results of model training\n");
    printf("Grind rate: %d\n", (int) ((NUM_TRAIN * EPOCHS) / total_train_time));
    printf("Total training time: %f seconds\n", total_train_time);
    printf("Total inference time: %f seconds\n", total_test_time);
    printf("Learning rate: %f\n", alpha);
    printf("Batch size: %d\n", m);
}