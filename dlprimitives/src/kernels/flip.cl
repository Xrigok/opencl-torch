///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
void flip_exec(__global dtype const *x,ulong  x_offset,
             __global dtype *y, ulong  y_offset,
             __global long const *DL,ulong DL_offset,
             __global int const *Dims,ulong Dims_offset){
    const long index = get_global_id(0);
    float index_map[L1];
    long M1=1,M2=1;
    for(int i=L1-1;i>=0;--i){
        M1 *= DL[DL_offset + i];
        index_map[i] = (index % M1) / M2;
        M2 *= DL[DL_offset + i];
    }
    for(int i=0;i<L2;++i){
        int idx = Dims[Dims_offset + i];
        index_map[idx] = DL[DL_offset + idx] - index_map[idx] - 1;
    }
    long out_index=0;
    long M3 = 1;
    for(int i=L1-1;i>=0;--i){
        out_index += index_map[i] * M3;
        M3 *= DL[DL_offset + i];
    }
    y[y_offset + out_index] = x[x_offset + index];

}