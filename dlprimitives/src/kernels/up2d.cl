///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
__kernel
void up2d_forward(__global dtype const *x, ulong x_offset,
             __global dtype *y, ulong y_offset,double s_h,double s_w){
    const int nc = get_global_id(0);
    const int h = get_global_id(1);
    const int w = get_global_id(2);
    if(h>=OH||w>=OW){
        return;
    }
    int xh=h/s_h;
    int xw=w/s_w;
    y[y_offset+nc*OH*OW+h*OW+w]=x[x_offset+nc*IH*IW+xh*IW+xw];
}