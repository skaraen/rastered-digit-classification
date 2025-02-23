#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mnist.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define EPOCHS 5

float init(int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    return ((float)rand() / RAND_MAX) * 2 * limit - limit;
}

void weighted_product_relu(float* res, float* in, float** w, float *b, int in_size, int out_size) {
    printf("W1[64][0]: %f\n", w[64][0]);
    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_size; j++) {
            printf("W1[%d][%d]: %f\n", i, j, w[i][j]);
            sum += w[i][j] * in[j];
        }
        res[i] = fmaxf(0, sum + b[i]);
    }
}

void weighted_product_sigmoid(float* res, float* in, float** w, float *b, int in_size, int out_size) {
    float sig_denom = 0.0f;
    float* sum = malloc(out_size * sizeof(float));
    for (int i = 0; i < out_size; i++) {
        sum[i] = 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum[i] += w[i][j] * in[j];
        }
        sig_denom += expf(sum[i] + b[i]);
    }

    for (int i = 0; i < out_size; i++) {
        res[i] = expf(sum[i] + b[i]) / sig_denom;
    }
    free(sum);
}

int main(int argc, char** argv) {
    //printf("Hello\n");
    int n_h1 = atoi(argv[1]);
    int n_h2 = atoi(argv[2]);
    float alpha = atof(argv[3]);
    int m = atoi(argv[4]);

    load_mnist();

    printf("in: %d, out: %d, n_h1: %d, n_h2: %d\n", INPUT_SIZE, OUTPUT_SIZE, n_h1, n_h2);

    float *h1 = malloc(n_h1 * sizeof(float));
    float *h2 = malloc(n_h2 * sizeof(float));

    float **w1 = malloc(n_h1 * sizeof(float));
    for (int i =0; i < n_h1; i++) {
        w1[i] = malloc(INPUT_SIZE * sizeof(float));
    }
    float **w2 = malloc(n_h2 * sizeof(float));
    for (int i =0; i < n_h2; i++) {
        w2[i] = malloc(n_h1 * sizeof(float));
    }
    float **w3 = malloc(OUTPUT_SIZE * sizeof(float));
    for (int i =0; i < OUTPUT_SIZE; i++) {
        w3[i] = malloc(n_h2 * sizeof(float));
    }

    float *b1 = malloc(n_h1 * sizeof(float));
    float *b2 = malloc(n_h2 * sizeof(float));
    float *b3 = malloc(OUTPUT_SIZE * sizeof(float));

    float *out = malloc(OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < n_h1; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            w1[i][j] = init(INPUT_SIZE, n_h1);
            // printf("W1[%d][%d]: %f\n", i, j, w1[i][j]);
        }
        b1[i] = 0;
    }

    for (int i = 0; i < n_h2; i++) {
        for (int j = 0; j < n_h1; j++) {
            w2[i][j] = init(n_h1, n_h2);
            // printf("W2[%d][%d]: %f\n", i, j, w2[i][j]);
        }
        b2[i] = 0;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < n_h2; j++) {
            w3[i][j] = init(n_h2, OUTPUT_SIZE);
            // printf("W3[%d][%d]: %f\n", i, j, w3[i][j]);
        }
        b3[i] = 0;
    }

    float *d_b1 = malloc(n_h1 * sizeof(float));
    float *d_b2 = malloc(n_h2 * sizeof(float));
    float *d_b3 = malloc(OUTPUT_SIZE * sizeof(float));

    float **d_w1 = malloc(n_h1 * sizeof(float));
    for (int i =0; i < n_h1; i++) {
        d_w1[i] = calloc(INPUT_SIZE, sizeof(float));
        d_b1[i] = 0.0f;
    }
    float **d_w2 = malloc(n_h2 * sizeof(float));
    for (int i =0; i < n_h2; i++) {
        d_w2[i] = calloc(n_h1, sizeof(float));
        d_b2[i] = 0.0f;
    }
    float **d_w3 = malloc(OUTPUT_SIZE * sizeof(float));
    for (int i =0; i < OUTPUT_SIZE; i++) {
        d_w3[i] = calloc(n_h2, sizeof(float));
        d_b3[i] = 0.0f;
    }

    printf("Initialization done, training starts...\n");
    float* error_h1 = (float*)malloc(n_h1 * sizeof(float));
    float* error_h2 = (float*)malloc(n_h2 * sizeof(float));

    //  Train
    for (int k = 0; k < m; k++) {
        printf("Training for batch %d\n", k + 1);
        float C = 0.0f;
        for (int x = k * m; x < (k + 1) * m; x++) {
            // Feed-forward
            weighted_product_relu(h1, train_image[x], w1, b1, INPUT_SIZE, n_h1);
            weighted_product_relu(h2, h1, w2, b2, n_h1, n_h2);
            weighted_product_sigmoid(out, h2, w3, b3, n_h2, OUTPUT_SIZE);

            printf("Feed forward completed %d, batch %d\n", x, k + 1);

            C += -log(out[train_label[x]]);

            // SGD calculation
            // Output layer
            float error_out[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                error_out[i] = (i == train_label[x]) ? out[i] - 1.0f : out[i]; 
            }

            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < n_h2; j++) {
                    d_w3[i][j] += error_out[i] * h2[j];
                }
                d_b3[i] += error_out[i];
            }

            // Hidden layer 2
            for (int i = 0; i < n_h2; i++) {
                float sum = 0.0f;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    sum += w3[j][i] * error_out[j];  
                }
                error_h2[i] = (h2[i] > 0) ? sum : 0.0f;
            }

            for (int i = 0; i < n_h2; i++) {
                for (int j = 0; j < n_h1; j++) {
                    d_w2[i][j] += error_h2[i] * h1[j];
                }
                d_b2[i] += error_h2[i];
            }

            // Hidden layer 1
            for (int i = 0; i < n_h1; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n_h2; j++) {
                    sum += w2[j][i] * error_h2[j];  
                }
                error_h1[i] = (h1[i] > 0) ? sum : 0.0f;
            }

            for (int i = 0; i < n_h1; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    d_w1[i][j] += error_h1[i] * test_image[(int) (j / 28)][(int) (j % 28)];
                }
                d_b1[i] += error_h1[i];
            }
        }

        printf("Batch %d / %d completed, Average cost = %f \n", k + 1, NUM_TRAIN / m, C / m);

        // Back propagation
        for (int i = 0; i < n_h1; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                w1[i][j] -= (alpha * d_w1[i][j]) / m;
                d_w1[i][j] = 0.0f;
            }
            b1[i] -= (alpha * d_b1[i]) / m;
            d_b1[i] = 0.0f;
        }

        for (int i = 0; i < n_h2; i++) {
            for (int j = 0; j < n_h1; j++) {
                w2[i][j] -= (alpha * d_w2[i][j]) / m;
                d_w2[i][j] = 0.0f;
            }
            b2[i] -= (alpha * d_b2[i]) / m;
            d_b2[i] = 0.0f;
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < n_h2; j++) {
                w3[i][j] -= (alpha * d_w3[i][j]) / m;
                d_w3[i][j] = 0.0f;
            }
            b3[i] -= (alpha * d_b3[i]) / m;
            d_b3[i] = 0.0f;
        }
    }

    for (int i =0; i < n_h1; i++) {
        free(w1[i]); 
        free(d_w1[i]);
    }
    free(w1);
    free(d_w1);

    for (int i =0; i < n_h2; i++) {
        free(w2[i]); 
        free(d_w2[i]);
    }
    free(w2);
    free(d_w2);

    for (int i =0; i < OUTPUT_SIZE; i++) {
        free(w3[i]); 
        free(d_w3[i]);
    }
    free(w3);
    free(d_w3);

    free(b1);
    free(b2);
    free(b3);

    free(d_b1);
    free(d_b2);
    free(d_b3);

    free(error_h1);
    free(error_h2);
}