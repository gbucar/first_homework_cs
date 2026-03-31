# First homework


## Task 1: Analyze performance of O3 processor (5 points)

### Task 1a:

| Metric       | MinorCPU | O3CPU    |
|--------------|----------|----------|
| CPI          | 1.509098 | 0.894233 |
| Total cycles | 100933   | 59809    |
| IPC          | 0.662647 | 1.118277 |

The O3CPU model achieves a $1.69\times$ higher IPC compared to the MinorCPU,
 demonstrating its superior efficiency in exploiting instruction-level parallelism.
This performance gain is primarily driven by out-of-order execution,
which allows the processor to dynamically schedule instructions.


### Task 1b: 

| Rob size   | 16       | 32       | 64       | 128      | 256       | 
|------------|----------|----------|----------|----------|-----------|
| O3         | 0.844347 | 1.006918 | 1.134746 | 1.141128 | 1.141128  |

![IPC vs ROB](./img/IPC_vs_ROB.png)

Performance saturates at a ROB size of 128, beyond which no further IPC gains are observed. This indicates that the workload's ILP is limited by intrinsic data dependencies, rendering larger instruction windows redundant for this specific SEQ_LEN.

### Task 1c:  

| width | IPC (original) | IPC (optimized) |
|-------|----------------|-----------------|
| 2     | 0.828108       | 0.796039        |
| 4     | 0.873591       | 0.831634        |
| 8     | 0.893775       | 0.835853        |

![orig vs optimized](./img/IPC_orig_vs_optimized.png)

The IPC gap between the two versions grows as the pipeline widens, showing that the original kernel scales more effectively with increased hardware resources.
This reveals that the optimized version likely contains more data dependencies or bottlenecks that limit instruction-level parallelism (ILP), preventing it from fully utilizing the additional execution width of the O3CPU.

### Task 1d:

**Original**
| Metric | 64       | 96       | 128      |
|--------|----------|----------|----------|
| IPC    | 1.107041 | 1.240941 | 1.231028 | 
| Stalls | 18047    | 8369     | 144      | 

**Optimized**
| Metric | 64       | 96       | 128      |
|--------|----------|----------|----------|
| IPC    | 1.376532 | 1.612539 | 1.551698 | 
| Stalls | 2747     | 44       | 0        | 

![Graph](./img/IPC_STALLS_vs_registers.png)

Both versions peak at 96 physical registers.

## Task 2: Branch Prediction and Speculative Execution in Masked Attention (O3CPU) (5 points)

### Task 2a

| Metric                        | Value   |
|-------------------------------|---------|
| Total instructions committed  | 2194513 |
| Total cycles                  | 1774059 |
| IPC                           | 1.23700 |
| Branch instructions committed | 335874  |
| Branch mispredictions         | 496     |


### Task 2b

| Predictor  | Branch mispredictions  | IPC     | 
|------------|------------------------|---------|
| TAGE       | 1088                   | 0.81348 |
| LocalBP    | 5430                   | 0.83041 |
| Tournament | 4673                   | 0.82353 |
| BimodeBP   | 4788                   | 0.82484 |

TAGE achieves the lowest misprediction count because it uses multiple tables with varying history lengths to accurately capture both simple and complex branching patterns.

### Task 2c

| ROB Size | Misprediction count | IPC      | Squashed instruction count |
|----------|---------------------|--------- |----------------------------|
| 32       | 1000                | 0.920332 | 941                        |
| 64       | 1028                | 1.230500 | 1010                       |
| 128      | 1088                | 1.229281 | 1026                       |

As the ROB size increases, IPC initially rises because the processor can find more parallel instructions to execute, but then it plateaus as other hardware limits are reached. Simultaneously, the number of squashed instructions per misprediction increases because a larger buffer allows more "wrong-path" instructions to enter the pipeline before the error is caught.
