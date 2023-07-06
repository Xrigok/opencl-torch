///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"

__kernel
void myexec(PARAMS){
    const int index=get_global_id(0);
    LOAD
    CALC
    SAVE
}