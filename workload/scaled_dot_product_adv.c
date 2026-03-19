#include <stdio.h>
#include <math.h>
#include <gem5/m5ops.h>


#define SEQ_LEN 64
#define D_K     64
#define SCALE   0.125f
#define NEG_INF -1e9f

float dot_product(float *q, float *k, int dim) {
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

    for (int i = 0; i < dim; i += 4) {
        r0 += q[i+0] * k[i+0];
        r1 += q[i+1] * k[i+1];
        r2 += q[i+2] * k[i+2];
        r3 += q[i+3] * k[i+3];
    }

    return (r0 + r1) + (r2 + r3);
}

void softmax_row(float *scores, float *output, int len) {
    // Max reduction — 4 independent chains, branchless
    float m0 = scores[0], m1 = scores[1],
          m2 = scores[2], m3 = scores[3];

    for (int i = 4; i < len; i += 4) {
        m0 = scores[i+0] > m0 ? scores[i+0] : m0;
        m1 = scores[i+1] > m1 ? scores[i+1] : m1;
        m2 = scores[i+2] > m2 ? scores[i+2] : m2;
        m3 = scores[i+3] > m3 ? scores[i+3] : m3;
    }

    float max_val = m0 > m1 ? m0 : m1;
    max_val = m2 > max_val ? m2 : max_val;
    max_val = m3 > max_val ? m3 : max_val;

    // Exp pass — all independent, O3CPU overlaps them in ROB
    for (int i = 0; i < len; i++)
        output[i] = expf((scores[i] - max_val) * SCALE);

    // Sum accumulation — 4 independent chains
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int i = 0; i < len; i += 4) {
        s0 += output[i+0];
        s1 += output[i+1];
        s2 += output[i+2];
        s3 += output[i+3];
    }
    float sum = (s0 + s1) + (s2 + s3);

    // Normalize — reciprocal trick, multiply instead of divide
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i += 4) {
        output[i+0] *= inv_sum;
        output[i+1] *= inv_sum;
        output[i+2] *= inv_sum;
        output[i+3] *= inv_sum;
    }
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
    for (int j = 0; j < SEQ_LEN; j++)
        scores[j] = dot_product(Q[0], K[j], D_K);

    softmax_row(scores, attn_weights, SEQ_LEN);
    
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    for (int i = 0; i < 4; i++)
        printf("attn[%d] = %.6f\n", i, attn_weights[i]);

    return 0;
}