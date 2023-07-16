# opencl-torch
the backend of torch supported by opencl

The project is based on artyom-beilis, link:https://github.com/artyom-beilis/pytorch_dlprim/tree/032a6933fed7e9e0cdcf0dbadf9c7f246804fa3b 

how to install you can follow the above link. 

I add some ops(as follows),but only forward, unsupport backward. 
|op|
|---|
|arange|
|argsort|
|bmm|
|gelu|
|index_tensor_out|
|layernorm|
|maxtrix_power|
|max_dim|
|max_pool2d_with_indices|
|nonzero|
|UpsamplingNearest2d|
|filp|
|unique|

# Test
We test the partial of CSwin and yolov5 in 3090RTX, the result as follow: 
| net            | acc   |time    | 
|----------------|-------|--------|
| CSwin tiny(our)| 80.2% | 46.17ms| 
| CSwin tiny(cuda)| 80.2% | 40.04ms|
| CSWin_base_224(our)|82.6%|79.86ms|
| CSWin_base_224(cuda)|82.5%|60.70ms|
| yolov5n(our) |-----|50.44ms| 
| yolov5n(cuda)|-----|50.68ms|
| yolov5l6(our) |-----|64.82ms| 
| yolov5nl6(cuda)|-----|63.41ms|

CSwin test used 1K images, yolov5 only test time [have same result at 4 images].
