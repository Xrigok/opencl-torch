///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
__kernel
void index_put(__global float* A,ulong  A_offset,
                    const __global float* B,ulong B_offset,
                     const __global long* DS, ulong DS_offset,
                     const __global long* XX, ulong XX_offset) {
    const ulong x=get_global_id(0);
    const ulong y=get_global_id(1);
    const ulong z=get_global_id(2);

    const int x1=get_local_id(0);
    const int y1=get_local_id(1);
    const int z1=get_local_id(2);
    int idx=x1*ls2*ls3+y1*ls3+z1;

    __local long LXX[XL];
    for(int i=0;i<(XL+ls-1)/ls;++i){
        int xxidx=ls*i+idx;
        if(xxidx<XL){
            LXX[xxidx]=XX[XX_offset+xxidx];
        }
    }

    __local long LDS[M2];
    for(int i=0;i<(M2+ls-1)/ls;++i){
        int dsxidx=ls*i+idx;
        if(dsxidx<M2){
            LDS[dsxidx]=DS[DS_offset+dsxidx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    if(x>=DA||y>=DB||z>=DC){
        return;
    }

    long last=M2-1, last_l=DC;
    ulong offset=0;
    while(last>=0){
        long p=LXX[last*L+y];

        offset+=p*last_l;
        last_l*=LDS[last];
        --last;
    }
    A[A_offset+x*D1+offset+z]=B[B_offset+x*DB*DC+y*DC+z];
    
}