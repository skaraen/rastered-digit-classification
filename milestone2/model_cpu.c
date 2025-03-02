#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include "omp.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define EPOCHS 5
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

void matrix_add(float** res, float** a, float** b, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
}

void matrix_difference(float** res, float** a, float** b, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            res[i][j] = a[i][j] - b[i][j];
        }
    }
}

// void matrix_multiply(float** res, float** a, float** b, int x, int y, int z) {
//     for (int i = 0; i < x; i++) {
//         for (int k = 0; k < z; k++) {
//             res[i][k] = 0.0f;
//             for (int j = 0; j < y; j++) {
//                 res[i][k] += a[i][j] * b[j][k];
//             }
//         }
//     }
// }

// void matrix_multiply_transpose1(float** res, float** a, float** b, int x, int y, int z) {
//     //printf("Henlo\n");
//     for (int i = 0; i < y; i++) {  
//         for (int k = 0; k < z; k++) {
//             res[i][k] = 0.0f;
//             for (int j = 0; j < x; j++) {  
//                 //printf("res[%d][%d] += a[%d][%d] * b[%d][%d]\n", i, k, j, i, j, k);
//                 res[i][k] += a[j][i] * b[j][k];
//             }
//         }
//     }
// }

// void matrix_multiply_transpose2(float** res, float** a, float** b, int x, int y, int z) {
//     for (int i = 0; i < x; i++) {
//         for (int k = 0; k < z; k++) {
//             res[i][k] = 0.0f;
//             for (int j = 0; j < y; j++) {
//                 res[i][k] += a[i][j] * b[k][j]; 
//             }
//         }
//     }
// }  

void tiled_matrix_multiply(float** res, float** a, float** b, int x, int y, int z) {
    #pragma omp parallel for 
    for (int i0 = 0; i0 < x; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < z; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < y; j0 += TILE_SIZE) {
                for (int i = i0; i < min(i0 + TILE_SIZE, x); i++) {
                    for (int k = k0; k < min(k0 + TILE_SIZE, z); k++) {
                        float sum = (j0 == 0) ? 0.0f : res[i][k];
                        for (int j = j0; j < min(j0 + TILE_SIZE, y); j++) {
                            sum += a[i][j] * b[j][k];
                        }
                        res[i][k] = sum;
                    }
                }
            }
        }
    }
}

void tiled_matrix_multiply_transpose1(float** res, float** a, float** b, int x, int y, int z) {
    #pragma omp parallel for 
    for (int i0 = 0; i0 < y; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < z; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < x; j0 += TILE_SIZE) {
                for (int i = i0; i < min(i0 + TILE_SIZE, y); i++) {
                    for (int k = k0; k < min(k0 + TILE_SIZE, z); k++) {
                        float sum = (j0 == 0) ? 0.0f : res[i][k];
                        for (int j = j0; j < min(j0 + TILE_SIZE, x); j++) {
                            sum += a[j][i] * b[j][k];
                        }
                        res[i][k] = sum;
                    }
                }
            }
        }
    }
}

void tiled_matrix_multiply_transpose2(float** res, float** a, float** b, int x, int y, int z) {
    #pragma omp parallel for 
    for (int i0 = 0; i0 < x; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < z; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < y; j0 += TILE_SIZE) {
                for (int i = i0; i < fmin(i0 + TILE_SIZE, x); i++) {
                    for (int k = k0; k < min(k0 + TILE_SIZE, z); k++) {
                        float sum = (j0 == 0) ? 0.0f : res[i][k];
                        for (int j = j0; j < min(j0 + TILE_SIZE, y); j++) {
                            sum += a[i][j] * b[k][j];
                        }
                        res[i][k] = sum;
                    }
                }
            }
        }
    }
}


void matrix_copy(float** res, float** a, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            res[i][j] = a[i][j];
        }
    }
}

void weighted_product_relu(float** res, float** w, float** a, float **b, int x, int y, int z) {
    tiled_matrix_multiply(res, w, a, x, y, z);
    matrix_add(res, res, b, x, z);

    for (int i = 0; i < x; i++) {
        for (int k = 0; k < z; k++) {
            res[i][k] = fmaxf(0, res[i][k]);
        }
    }
}

void weighted_product_sigmoid(float** res, float** w, float** a, float **b, int x, int y, int z) {
    tiled_matrix_multiply(res, w, a, x, y, z);
    matrix_add(res, res, b, x, z);

    float sum[z];
    for (int k = 0; k < z; k++) {
        sum[k] = 0.0f;
        for (int i = 0; i < x; i++) {
            sum[k] += expf(res[i][k]);
        }
    }

    for (int i = 0; i < x; i++) {
        for (int k = 0; k < z; k++) {
            res[i][k] = expf(res[i][k]) / sum[k];
        }
    }
}

float** init_2D(int m, int n) {
    float **mat = malloc(m * sizeof(float*));
    for (int i = 0; i < m; i++) {
        mat[i] = malloc(n * sizeof(float));
    }

    return mat;
}

float** init_xavier(int m, int n) {
    float **mat = init_2D(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = xavier(m, n);
        }
    }

    return mat;
}

float** init_zero(int m, int n) {
    float **mat = init_2D(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = 0.0f;
        }
    }

    return mat;
}

void update_weights(float **w, float **d_w, int x, int y, float alpha, int m) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            w[i][j] -= (alpha * d_w[i][j]) / m;
            d_w[i][j] = 0.0f;
        }
    }
}

void update_biases(float **b, float **d_b, int x, int y, float alpha, int m) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            b[i][j] -= (alpha * d_b[i][j]);
            d_b[i][j] = 0.0f;
        }
    }
}

void reset(float **a, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            a[i][j] = 0.0f;
        }
    }
}

int main(int argc, char** argv) {
    //printf("Hello\n");
    int n_h1 = atoi(argv[1]);
    int n_h2 = atoi(argv[2]);
    float alpha = atof(argv[3]);
    int m = atoi(argv[4]);
    int nthreads;

    #pragma omp parallel 
    {
        nthreads = omp_get_num_threads();
    }

    load_mnist();

    // Initialize network layers
    float **in = init_2D(INPUT_SIZE, m);
    float **h1 = init_2D(n_h1, m);
    float **h2 = init_2D(n_h2, m);
    float **out = init_2D(OUTPUT_SIZE, m);

    // Initialize weights
    float **w1 = init_xavier(n_h1, INPUT_SIZE);
    float **w2 = init_xavier(n_h2, n_h1);
    float **w3 = init_xavier(OUTPUT_SIZE, n_h2);

    // Initialize biases
    float **b1 = init_zero(n_h1, m);
    float **b2 = init_zero(n_h2, m);
    float **b3 = init_zero(OUTPUT_SIZE, m);

    // Initialize delta weights and biases
    float **d_w1 = init_zero(n_h1, INPUT_SIZE);
    float **d_w2 = init_zero(n_h2, n_h1);
    float **d_w3 = init_zero(OUTPUT_SIZE, n_h2);

    float **d_b1 = init_zero(n_h1, m);
    float **d_b2 = init_zero(n_h2, m);
    float **d_b3 = init_zero(OUTPUT_SIZE, m);

    float **error_out = init_zero(OUTPUT_SIZE, m);
    float **error_h2 = init_zero(n_h2, m);
    float **error_h1 = init_zero(n_h1, m);

    float **y = init_zero(OUTPUT_SIZE, m);
    float **y_h2 = init_zero(n_h2, m);
    float **y_h1 = init_zero(n_h1, m);

    printf("Initialization done, training starts...\n");

    float total_train_time, total_test_time = 0.0f;
    double start_time, end_time;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float C_train = 0.0f;
        float C_test = 0.0f;
        shuffle_data(&train_image[0][0], train_label, NUM_TRAIN, INPUT_SIZE);

        //printf("Images shuffled\n");

        // Train network
        start_time = omp_get_wtime();
        for (int k = 0; k < NUM_TRAIN / m; k++) {
            float C = 0.0f;

            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i][x] = train_image[(k * m) + x][i];
                }
            }

            // Feed-forward
            weighted_product_relu(h1, w1, in, b1, n_h1, INPUT_SIZE, m);
            weighted_product_relu(h2, w2, h1, b2, n_h2, n_h1, m);
            weighted_product_sigmoid(out, w3, h2, b3, OUTPUT_SIZE, n_h2, m);

            for (int x = 0; x < m; x++) {
                int id = k * m + x;
                C += -log(out[train_label[id]][x]);
            }

            // SGD calculation
            // Output layer
            for (int x = 0; x < m; x++) {
                int id = k * m + x;
                y[train_label[id]][x] = 1.0f;
            }

            matrix_difference(error_out, out, y, OUTPUT_SIZE, m);
            matrix_copy(d_b3, error_out, OUTPUT_SIZE, m);
            tiled_matrix_multiply_transpose2(d_w3, error_out, h2, OUTPUT_SIZE, m, n_h2);

            // Hidden layer 2
            tiled_matrix_multiply_transpose1(y_h2, w3, error_out, OUTPUT_SIZE, n_h2, m);
            for (int i = 0; i < n_h2; i++) {
                for (int j = 0; j < m; j++) {
                    error_h2[i][j] = (h2[i][j] > 0) ? y_h2[i][j] : 0;
                }
            }
            tiled_matrix_multiply_transpose2(d_w2, error_h2, h1, n_h2, m, n_h1);
            matrix_copy(d_b2, error_h2, n_h2, m);

            // Hidden layer 1
            tiled_matrix_multiply_transpose1(y_h1, w2, error_h2, n_h2, n_h1, m);
            for (int i = 0; i < n_h1; i++) {
                for (int j = 0; j < m; j++) {
                    error_h1[i][j] = (h1[i][j] > 0) ? y_h1[i][j] : 0;
                }
            }
            tiled_matrix_multiply_transpose2(d_w1, error_h1, in, n_h1, m, INPUT_SIZE);
            matrix_copy(d_b1, error_h1, n_h1, m);

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
        total_train_time += (float) (end_time - start_time);

        // Testing
        start_time = omp_get_wtime();
        int fail_ct = 0;
        for (int k = 0; k < NUM_TEST / m; k++) {
            for (int x = 0; x < m; x++) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    in[i][x] = test_image[k * m + x][i];
                }
            }

            weighted_product_relu(h1, w1, in, b1, n_h1, INPUT_SIZE, m);
            weighted_product_relu(h2, w2, h1, b2, n_h2, n_h1, m);
            weighted_product_sigmoid(out, w3, h2, b3, OUTPUT_SIZE, n_h2, m);

            for (int x = 0; x < m; x++) {
                int prediction;
                float max_act = 0.0f;
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    if (max_act < out[i][x]) {
                        max_act = out[i][x];
                        prediction = i;
                    }
                }

                if (prediction != test_label[k * m + x]) {
                    fail_ct++;
                }
            }
        }
        end_time = omp_get_wtime();
        total_test_time += (float) (end_time - start_time);

        printf("Epoch %d / %d completed, Average train loss = %f, Failed count: %d, Test accuracy: %f\n", epoch + 1, EPOCHS, C_train / NUM_TRAIN, fail_ct, ((float) (NUM_TEST - fail_ct) / NUM_TEST) * 100.0f);
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        free(in[i]);
    }
    free(in);

    for (int i =0; i < n_h1; i++) {
        free(w1[i]); 
        free(d_w1[i]);
        free(h1[i]);
        free(b1[i]);
        free(d_b1[i]);
        free(error_h1[i]);
        free(y_h1[i]);
    }
    free(w1);
    free(d_w1);
    free(h1);
    free(b1);
    free(d_b1);
    free(error_h1);
    free(y_h1);

    for (int i =0; i < n_h2; i++) {
        free(w2[i]); 
        free(d_w2[i]);
        free(h2[i]);
        free(b2[i]);
        free(d_b2[i]);
        free(error_h2[i]);
        free(y_h2[i]);
    }
    free(w2);
    free(d_w2);
    free(h2);
    free(b2);
    free(d_b2);
    free(error_h2);
    free(y_h2);

    for (int i =0; i < OUTPUT_SIZE; i++) {
        free(w3[i]); 
        free(d_w3[i]);
        free(out[i]);
        free(b3[i]);
        free(d_b3[i]);
        free(error_out[i]);
        free(y[i]);
    }
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
    printf("Number of threads: %d\n", nthreads);
}