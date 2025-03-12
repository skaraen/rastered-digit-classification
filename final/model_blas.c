#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include <cblas.h>
#include <omp.h>

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define EPOCHS 50
#define TILE_SIZE 32 

#define min(a,b) ((a) < (b) ? (a) : (b))

float xavier(int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    return ((float)rand() / RAND_MAX) * 2 * limit - limit;
}

// Inspired from Fischer Yates shuffle algorithm
void shuffle_data(float* images, int* labels, int num_images, int img_dim) {
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

void matrix_add(float* res, float* a, float* b, int x, int y) {
    #pragma omp parallel for 
    for (int i = 0; i < x * y; i++) {
        res[i] = a[i] + b[i];
    }
}

void matrix_difference(float* res, float* a, float* b, int x, int y) {
    #pragma omp parallel for 
    for (int i = 0; i < x * y; i++) {
        res[i] = a[i] - b[i];
    }
}

void matrix_multiply_cblas(float* res, float* a, float* b, int x, int y, int z) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x, z, y, 1.0f, a, y, b, z, 0.0f, res, z);
}

void matrix_multiply_transpose1_cblas(float *res, float *a, float *b, int x, int y, int z) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, x, z, y, 1.0f, a, x, b, z, 0.0f, res, z);
}

void matrix_multiply_transpose2_cblas(float *res, float *a, float *b, int x, int y, int z) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x, z, y, 1.0f, a, y, b, y, 0.0f, res, z);
}

void tiled_matrix_multiply_transpose1(float* res, float* a, float* b, int x, int y, int z) {
    #pragma omp parallel for 
    for (int i0 = 0; i0 < y; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < z; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < x; j0 += TILE_SIZE) {
                for (int i = i0; i < min(i0 + TILE_SIZE, y); i++) {
                    for (int k = k0; k < min(k0 + TILE_SIZE, z); k++) {
                        float sum = (j0 == 0) ? 0.0f : res[i * z + k];
                        for (int j = j0; j < min(j0 + TILE_SIZE, x); j++) {
                            sum += a[j * y + i] * b[j * z + k];
                        }
                        res[i * z + k] = sum;
                    }
                }
            }
        }
    }
}

void tiled_matrix_multiply_transpose2(float* res, float* a, float* b, int x, int y, int z) {
    #pragma omp parallel for 
    for (int i0 = 0; i0 < x; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < z; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < y; j0 += TILE_SIZE) {
                for (int i = i0; i < min(i0 + TILE_SIZE, x); i++) {
                    for (int k = k0; k < min(k0 + TILE_SIZE, z); k++) {
                        float sum = (j0 == 0) ? 0.0f : res[i * z + k];
                        for (int j = j0; j < min(j0 + TILE_SIZE, y); j++) {
                            sum += a[i * y + j] * b[k * y + j];
                        }
                        res[i * z + k] = sum;
                    }
                }
            }
        }
    }
}

void matrix_copy(float* res, float* a, int x, int y) {
    for (int i = 0; i < x * y; i++) {
        res[i] = a[i];
    }
}

void weighted_product_relu(float* res, float* w, float* a, float* b, int x, int y, int z) {
    matrix_multiply_cblas(res, w, a, x, y, z);
    matrix_add(res, res, b, x, z);
    #pragma omp parallel for 
    for (int i = 0; i < x * z; i++) {
        res[i] = fmaxf(0, res[i]);
    }
}

void weighted_product_softmax(float* res, float* w, float* a, float* b, int x, int y, int z) {
    matrix_multiply_cblas(res, w, a, x, y, z);
    matrix_add(res, res, b, x, z);

    float sum[z];
    #pragma omp parallel for
    for (int k = 0; k < z; k++) {
        sum[k] = 0.0f;
        for (int i = 0; i < x; i++) {
            sum[k] += expf(res[i * z + k]);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < x; i++) {
        for (int k = 0; k < z; k++) {
            res[i * z + k] = expf(res[i * z + k]) / sum[k];
        }
    }
}

float* init_1D(int m, int n) {
    return (float*)calloc(m * n, sizeof(float));
}

float* init_xavier(int m, int n) {
    float* mat = init_1D(m, n);
    #pragma omp parallel for 
    for (int i = 0; i < m * n; i++) {
        mat[i] = xavier(m, n);
    }
    return mat;
}

float* init_zero(int m, int n) {
    float* mat = init_1D(m, n);
    #pragma omp parallel for 
    for (int i = 0; i < m * n; i++) {
        mat[i] = 0.0f;
    }
    return mat;
}

void update_weights(float* w, float* d_w, int x, int y, float alpha, int m) {
    #pragma omp parallel for 
    for (int i = 0; i < x * y; i++) {
        w[i] -= (alpha * d_w[i]) / m;
        d_w[i] = 0.0f;
    }
}

void reset(float *a, int x, int y) {
    #pragma omp parallel for 
    for (int i = 0; i < x * y; i++) {
        a[i] = 0.0f;
    }
}

int main(int argc, char** argv) {
    int n_h1 = atoi(argv[1]);
    int n_h2 = atoi(argv[2]);
    float alpha = atof(argv[3]);
    int m = atoi(argv[4]);

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
    float *d_w1 = init_zero(n_h1, INPUT_SIZE);
    float *d_w2 = init_zero(n_h2, n_h1);
    float *d_w3 = init_zero(OUTPUT_SIZE, n_h2);

    float *d_b1 = init_zero(n_h1, m);
    float *d_b2 = init_zero(n_h2, m);
    float *d_b3 = init_zero(OUTPUT_SIZE, m);

    float *error_out = init_zero(OUTPUT_SIZE, m);
    float *error_h2 = init_zero(n_h2, m);
    float *error_h1 = init_zero(n_h1, m);

    float *y = init_zero(OUTPUT_SIZE, m);
    float *y_h2 = init_zero(n_h2, m);
    float *y_h1 = init_zero(n_h1, m);

    printf("Initialization done, training starts...\n");

    double total_train_time = 0.0, total_test_time = 0.0;
    double start_time, end_time;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float C_train = 0.0f;
        float C_valid = 0.0f;
        shuffle_data(&train_image[0][0], train_label, NUM_TRAIN, INPUT_SIZE);

        //printf("Images shuffled\n");

        // Train network
        start_time = omp_get_wtime();
        for (int k = 0; k < NUM_TRAIN / m; k++) {
            float C = 0.0f;

            #pragma omp parallel for 
            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i * m + x] = train_image[(k * m) + x][i];
                }
            }

            // Feed-forward
            weighted_product_relu(h1, w1, in, b1, n_h1, INPUT_SIZE, m);
            weighted_product_relu(h2, w2, h1, b2, n_h2, n_h1, m);
            weighted_product_softmax(out, w3, h2, b3, OUTPUT_SIZE, n_h2, m);
            
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
            matrix_copy(d_b3, error_out, OUTPUT_SIZE, m);
            matrix_multiply_transpose2_cblas(d_w3, error_out, h2, OUTPUT_SIZE, m, n_h2);

            // Hidden layer 2
            matrix_multiply_transpose1_cblas(y_h2, w3, error_out, n_h2, OUTPUT_SIZE, m);
            #pragma omp parallel for 
            for (int i = 0; i < n_h2 * m; i++) {
                error_h2[i] = (h2[i] > 0) ? y_h2[i] : 0;
                d_b2[i] = error_h2[i];
            }
            matrix_multiply_transpose2_cblas(d_w2, error_h2, h1, n_h2, m, n_h1);

            // Hidden layer 1
            matrix_multiply_transpose1_cblas(y_h1, w2, error_h2, n_h1, n_h2, m);
            #pragma omp parallel for 
            for (int i = 0; i < n_h1 * m; i++) {
                error_h1[i] = (h1[i] > 0) ? y_h1[i] : 0;
                d_b1[i] = error_h1[i];
            }
            matrix_multiply_transpose2_cblas(d_w1, error_h1, in, n_h1, m, INPUT_SIZE);

            // Back propagation
            update_weights(w1, d_w1, n_h1, INPUT_SIZE, alpha, m);
            update_weights(w2, d_w2, n_h2, n_h1, alpha, m);
            update_weights(w3, d_w3, OUTPUT_SIZE, n_h2, alpha, m);

            update_weights(b1, d_b1, n_h1, m, alpha, m);
            update_weights(b2, d_b2, n_h2, m, alpha, m);
            update_weights(b3, d_b3, OUTPUT_SIZE, m, alpha, m);

            C_train += C;
            reset(y, OUTPUT_SIZE, m);
        }
        end_time = omp_get_wtime();
        total_train_time += end_time - start_time;

        // Validation
        for (int k = 0; k < NUM_VALID / m; k++) {
            float C = 0.0f;
            #pragma omp parallel for
            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i * m + x] = valid_image[k * m + x][i];
                }
            }

            weighted_product_relu(h1, w1, in, b1, n_h1, INPUT_SIZE, m);
            weighted_product_relu(h2, w2, h1, b2, n_h2, n_h1, m);
            weighted_product_softmax(out, w3, h2, b3, OUTPUT_SIZE, n_h2, m);

            for (int x = 0; x < m; x++) {
                int id = k * m + x;
                C += -log(out[valid_label[id] * m + x]);
            }
            C_valid += C;
        }

        // Testing
        int fail_ct = 0;
        start_time = omp_get_wtime();
        for (int k = 0; k < NUM_TEST / m; k++) {
            for (int x = 0; x < m; x++) {
                #pragma omp parallel for 
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i * m + x] = test_image[k * m + x][i];
                }
            }

            weighted_product_relu(h1, w1, in, b1, n_h1, INPUT_SIZE, m);
            weighted_product_relu(h2, w2, h1, b2, n_h2, n_h1, m);
            weighted_product_softmax(out, w3, h2, b3, OUTPUT_SIZE, n_h2, m);

            for (int x = 0; x < m; x++) {
                int prediction;
                float max_act = 0.0f;
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    if (max_act < out[i * m + x]) {
                        max_act = out[i * m + x];
                        prediction = i;
                    }
                }

                if (prediction != test_label[k * m + x]) {
                    fail_ct++;
                }
            }
        }
        end_time = omp_get_wtime();
        total_test_time += end_time - start_time;

        printf("Epoch %d / %d, Avg. training loss = %f,  Avg. validation loss = %f, Test accuracy: %f\n", epoch + 1, EPOCHS, C_train / NUM_TRAIN, C_valid / NUM_VALID, ((float) (NUM_TEST - fail_ct) / NUM_TEST) * 100.0f);
    }

    free(in);
    free(w1);
    free(d_w1);
    free(h1);
    free(b1);
    free(d_b1);
    free(error_h1);
    free(y_h1);
    free(w2);
    free(d_w2);
    free(h2);
    free(b2);
    free(d_b2);
    free(error_h2);
    free(y_h2);
    free(w3);
    free(d_w3);
    free(out);
    free(b3);
    free(d_b3);
    free(error_out);
    free(y);

    printf("Results of model training\n");
    printf("Grind rate: %d\n", (int) ((NUM_TRAIN * EPOCHS) / total_train_time));
    printf("Total training time: %f seconds\n", total_train_time);
    printf("Total inference time: %f seconds\n", total_test_time);
    printf("Learning rate: %f\n", alpha);
    printf("Batch size: %d\n", m);
}
