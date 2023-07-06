///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////

#define eps 1e-5

__kernel
void forward_affine(__global float const *x,ulong  x_offset,
             __global float *y,      ulong  y_offset,
             __global float const *A,ulong A_offset,
             __global float const *B,ulong B_offset){
    int gidx=get_global_id(0);
    int idx  = get_local_id(0);
    int offset=get_global_id(0)/local_size*norm_size;
    __local float subx[norm_size];

    __local float exs[local_size];
    __local float xqs[local_size];

    float ex=0,xq=0;

    for(int i=0;i<(norm_size+local_size-1)/local_size;++i){
        if(idx+local_size*i<norm_size){
            subx[idx+local_size*i]=x[x_offset+offset+idx+local_size*i];
            ex+=subx[idx+local_size*i];
            xq+=subx[idx+local_size*i]*subx[idx+local_size*i];
        }
    }

    exs[idx]=ex;
    xqs[idx]=xq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int s2=2; s2<=local_size; s2<<=1){
        if((idx&(s2-1))==0){
            exs[idx]+=exs[idx+(s2>>1)];
            xqs[idx]+=xqs[idx+(s2>>1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    ex=exs[0]/norm_size;
    xq=xqs[0]/norm_size;
    float div=1.f/sqrt(xq-ex*ex+eps);
    
    for(int i=0;i<(norm_size+local_size-1)/local_size;++i){
        if(idx+local_size*i<norm_size){
            y[x_offset+offset+idx+local_size*i]=(subx[idx+local_size*i]-ex)*A[A_offset+idx+local_size*i]*div+ B[B_offset + idx+local_size*i];
        }
    }

}

__kernel
void forward_direct( __global float const *x,ulong  x_offset,
             __global float *y,      ulong  y_offset)
{
    int idx  = get_local_id(0);
    int offset=get_global_id(0)/local_size*norm_size;
    __local float subx[norm_size];

    __local float exs[local_size];
    __local float xqs[local_size];

    float ex=0,xq=0;

    for(int i=0;i<(norm_size+local_size-1)/local_size;++i){
        if(idx+local_size*i<norm_size){
            subx[idx+local_size*i]=x[x_offset+offset+idx+local_size*i];
            ex+=subx[idx+local_size*i];
            xq+=subx[idx+local_size*i]*subx[idx+local_size*i];
        }
    }
    exs[idx]=ex;
    xqs[idx]=xq;
    barrier(CLK_LOCAL_MEM_FENCE);

    int s1=1,s2=2;
    while(s2<=local_size){
        if((idx&(s2-1))==0){
            exs[idx]+=exs[idx+s1];
            xqs[idx]+=xqs[idx+s1];
            printf("%d:%d\n",idx,idx+s1);
        }
        s1=s2;
        s2*=2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    ex=exs[0]/norm_size;
    xq=xqs[0]/norm_size;
    float div=sqrt(xq-ex*ex+eps);

    for(int i=0;i<(norm_size+local_size-1)/local_size;++i){
        if(idx+local_size*i<norm_size){
            y[x_offset+offset+idx+local_size*i]=(subx[idx+local_size*i]-ex)/div;
        }
    }

}