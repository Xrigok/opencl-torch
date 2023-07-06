///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////

bool compare(float a,float b){
    if(a<b){
        return false;
    }
    return true;
}
bool compareStable(float a ,long idxa,float b, long idxb){
    if(a<b){
        return false;
    }
    if(a==b&&idxa<idxb){
        return false;
    }
    return true;
}

void quicksort(float *a,long *idxs,int left,int right){
    if(left>=right){
        return;
    }
    int left1=left,right1=right;
    while(left1<right1){
        while(left1<right1&&CODE){
            --right1;
        }
        float temp1=a[left1];
        a[left1]=a[right1];
        a[right1]= temp1;

        ulong temp2=idxs[left1];
        idxs[left1]=idxs[right1];
        idxs[right1]=temp2;

        while(left1<right1&&CODE){
            ++left1;
        }
        float temp3=a[left1];
        a[left1]=a[right1];
        a[right1]= temp3;

        ulong temp4=idxs[left1];
        idxs[left1]=idxs[right1];
        idxs[right1]=temp4;
    }
    quicksort(a,idxs,left,left1-1);
    quicksort(a,idxs,left1+1,right);
}

__kernel
void quick_sort(__global float const *A,ulong  A_offset,
             __global float *B,      ulong  B_offset,
             __global long *C,        ulong  C_offset)
{
    int x=get_global_id(0);
    int z=get_global_id(2);
    int ly=get_local_id(1);
    __local float v[M2];
    __local long idxs[M2];
    
    for(long i=0;i<(M2+LS-1)/LS;++i){
        long idx=LS*i+ly;
        if(idx<M2){
            v[idx]=A[A_offset+x*M2*M3+idx*M3+z];
            idxs[idx]=idx;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ly==0){
        quicksort(v,idxs,0,M2-1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    #if descending
        for(long i=0;i<(M2+LS-1)/LS;++i){
            long idx=LS*i+ly;
            if(idx<M2){
                B[B_offset+x*M2*M3+idx*M3+z]=v[idx];
                C[C_offset+x*M2*M3+idx*M3+z]=idxs[idx];;
            }
        }
    #else
        for(long i=0;i<(M2+LS-1)/LS;++i){
            long idx=LS*i+ly;
            if(idx<M2){
                B[B_offset+x*M2*M3+idx*M3+z]=v[M2-1-idx];
                C[C_offset+x*M2*M3+idx*M3+z]=idxs[M2-1-idx];;
            }
        }
    #endif
}