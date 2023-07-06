///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
void maxpoolingindice_forward(__global dtype const *x, ulong x_offset,
             __global dtype *y, ulong y_offset,
             __global ulong *inds, ulong inds_offset){
    int onc = get_global_id(0);
    int oh = get_global_id(1);
    int ow = get_global_id(2);

    if(oh>=OH||ow>=OW){
        return;
    }
    int lt_x=-px+sx*oh;
    int lt_y=-py+sy*ow;

    int I_offset=onc*IH*IW;
    int O_offset=onc*OH*OW;
    int max_ind=0;

    dtype maxv=-65535;
    if(lt_x>=0&&lt_x<IH&&lt_y>=0&&lt_y<IW){
        maxv=x[I_offset+lt_x*IW+lt_y];
        max_ind=lt_x*IW+lt_y;
    }
    dtype v=0;
    for(int cur_x=lt_x;cur_x<lt_x+kx*dx;cur_x+=dx){
        for(int cur_y=lt_y;cur_y<lt_y+ky*dy;cur_y+=dy){
            if(cur_x>=0&&cur_x<IH&&cur_y>=0&&cur_y<IW){
                v=x[x_offset+I_offset+cur_x*IW+cur_y];
                if(v>maxv){
                    maxv=v;
                    max_ind=cur_x*IW+cur_y;
                }
            }
        }
    }
    y[y_offset+O_offset+oh*OW+ow]=maxv;
    
    inds[inds_offset+O_offset+oh*OW+ow]=max_ind;
    
}