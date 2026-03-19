#include <stdio.h>
#include <math.h>
#include <gem5/m5ops.h>

#define SEQ_LEN 64
#define D_K 64
#define SCALE 0.125f
#define NEG_INF -1e9f

// Causal mask: position i can only attend to positions j <= i
void masked_softmax(float *scores, float *output, int query_pos, int len) {

    // Step 1: Apply causal mask — data-dependent branch per element
    for (int j = 0; j < len; j++) {
        if (j > query_pos)          // branch: taken ~50% for mid-sequence queries
            scores[j] = NEG_INF;
    }

    // Step 2: max reduction (same as before)
    float max_val = scores[0];
    for (int i = 1; i < len; i++) {
        if (scores[i] > max_val)    // branch: unpredictable, data-dependent
            max_val = scores[i];
    }

    // Step 3: exp + sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        output[i] = expf((scores[i] - max_val) * SCALE);
        sum += output[i];
    }

    // Step 4: normalize
    for (int i = 0; i < len; i++) {
        output[i] /= sum;
    }
}

float dot_product(float *q, float *k, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; i++)
        result += q[i] * k[i];
    return result;
}

int main() {
    float Q[SEQ_LEN][D_K];
    float K[SEQ_LEN][D_K];
    float scores[SEQ_LEN];
    float attn_weights[SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < D_K; j++) {
            Q[i][j] = (float)(i + j) * 0.01f;
            K[i][j] = (float)(i - j) * 0.01f;
        }
    
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif

    // Run masked softmax for every query position
    // This varies the branch taken/not-taken ratio across iterations
    for (int q = 0; q < SEQ_LEN; q++) {
        for (int j = 0; j < SEQ_LEN; j++)
            scores[j] = dot_product(Q[q], K[j], D_K);
        masked_softmax(scores, attn_weights, q, SEQ_LEN);
    }
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    for (int i = 0; i < 4; i++)
        printf("attn[%d] = %.6f\n", i, attn_weights[i]);

    return 0;
}