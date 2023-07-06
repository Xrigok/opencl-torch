///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////

__kernel
void dim_max_min(__global float const *A,ulong  A_offset,
             __global float *B,      ulong  B_offset,
             __global ulong *C,        ulong  C_offset)
{
    int x=get_global_id(0);
    int z=get_global_id(2);
    int gy=get_group_id(1);
    int ly=get_local_id(1);
    __local float v[P];
    __local int idx[P];
    v[ly]=A[A_offset+x*L1*L2+(gy*P+ly)*L2+z];
    idx[ly]=ly;
    for(int i=1;i<(L1+P-1)/P;++i){
        if(P*i+ly<L1){
            int offset=x*L1*L2+(gy*P+P*i+ly)*L2+z;
            float cur_v=A[A_offset+offset];
            if(CALC1){
                v[ly]=cur_v;
                idx[ly]=P*i+ly;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int mid=P/2;mid>0;mid/=2){
        if(ly<mid&&v[ly]<v[ly+mid]){
            v[ly]=v[ly+mid];
            idx[ly]=idx[ly+mid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ly==0){
        B[B_offset+x*L2+z]=v[0];
        C[C_offset+x*L2+z]=idx[0];
    }
    

}