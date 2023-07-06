///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
__kernel
void bmm_forward(const __global float* A,ulong  A_offset,
                     const __global float* B,ulong B_offset,
                     __global float* C,ulong C_offset,
                     int M1, int M2, int K) {

    const int row = get_local_id(0);//0...P-1
    const int col = get_local_id(1);//0...P-1
    const int globalRow = get_group_id(0)*P+row;//0...M1
    const int globalCol = get_group_id(1)*P+col;//0...M2
    const int i = get_group_id(2);

    __local float Asub[P][P];
    __local float Bsub[P][P];
    __local float Csub[P][P];
    
    const int numTiles=(K+P-1)/P;

    Csub[row][col]=0.f;
    const int offset1=i*M1*K;
    const int offset2=i*K*M2;
    const int offset3=i*M1*M2;
    for(int t=0;t<numTiles;++t){
        
        Asub[row][col]=(globalRow>=0&&globalRow<M1&&col+t*P>=0&&col+t*P<K)?A[A_offset+offset1+globalRow*K+col+t*P]:0.f;
        Bsub[col][row]=(row+t*P>=0&&row+t*P<K&&globalCol>=0&&globalCol<M2)?B[B_offset+offset2+(row+t*P)*M2+globalCol]:0.f;

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int j=0; j<P; ++j){
            Csub[row][col]+=Asub[row][j]*Bsub[col][j];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    if(globalRow<M1&&globalCol<M2){
        C[C_offset+offset3+globalRow*M2+globalCol]=Csub[row][col];
    }
    
}
