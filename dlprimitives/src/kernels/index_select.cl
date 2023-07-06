///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
void index_select(const __global float* A,ulong  A_offset,
                    __global float* B,ulong B_offset,
                     const __global long* DS, ulong DS_offset,
                     const __global long* XX, ulong XX_offset) {
    const ulong idx=get_global_id(0);
    
    const int lidx=get_local_id(0);
    __local long LXX[XL];
    for(int i=0;i<(XL+ls-1)/ls;++i){
        int xxidx=ls*i+lidx;
        if(xxidx<XL){
            LXX[xxidx]=XX[XX_offset+xxidx];
        }
    }

    __local long LDS[M2];
    for(int i=0;i<(M2+ls-1)/ls;++i){
        int dsxidx=ls*i+lidx;
        if(dsxidx<M2){
            LDS[dsxidx]=DS[DS_offset+dsxidx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(idx>=out_size){
        return;
    }

    ulong idx1=idx/OD1;
    ulong offset=idx1*D1;

    ulong idx2=(idx%OD1)/D2;
    long last=M2-1, last_l=D2;
    
    while(last>=0){
        long p=LXX[last*L+idx2];

        offset+=p*last_l;
        last_l*=LDS[last];
        --last;
    }

    #if M3
        B[B_offset+idx]=A[A_offset+offset+idx%D2];
    #else
        B[B_offset+idx]=A[A_offset+offset];
    #endif
    
}
