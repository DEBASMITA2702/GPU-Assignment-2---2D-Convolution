# GPU Programming – Assignment 2

## Problem Statement
Implementing a **2D Convolution operation** using CUDA on stacked 2D matrices.  
The task involves applying filters on images with GPU parallelism using:
- Shared memory  
- Memory coalescing  
- Zero-padding (stride = 1)  

---

## Input & Output
- **Input Image:** `H × W × C` → transformed into `(H × C) × W`  
- **Filter Set:** `K × R × S × C` → each filter as `(R × C) × S`  
- **Output:** For each filter, output is `H × W` (stacked vertically if multiple filters)  

---

### Example
Image Dimensions: 3 3 1  
Image Matrix:
```
1 2 3
4 5 6
7 8 9
```

Filter Dimensions: 1 3 3 1  
Filter Matrix:
```
1 1 1
1 1 1
1 1 1
```

Output Matrix:
```
12 21 16
27 45 33
24 39 28
```

---

## Execution
nvcc <convolution>.cu -o conv2d  
./conv2d


