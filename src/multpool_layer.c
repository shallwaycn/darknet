#include "multpool_layer.h"
#include "cuda.h"
#include <stdio.h>



multpool_layer make_multpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int extra)
{
    multpool_layer l = {0};
    l.type = MULTPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    if (extra == 1){
        l.forward = forward_multpool_layer1;
        l.backward = backward_multpool_layer1;
    }
    else{
        l.forward = forward_multpool_layer2;
        l.backward = backward_multpool_layer2;
    }
    #ifdef GPU
    l.forward_gpu = forward_multpool_layer_gpu;
    l.backward_gpu = backward_multpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "mult          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_multpool_layer(multpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void forward_multpool_layer1(const multpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    float mult = 0;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;

                            // float val_mult = (valid != 0) ? net.input[index] : -FLT_MAX;
                            // mult += val_mult;
                        }
                    }
                    //l.output[out_index] = mult/(l.size*l.size);
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_multpool_layer1(const multpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }

    // int i,j,k,b;
    // for(b = 0; b < l.batch; ++b){
    //     for(k = 0; k < l.c; ++k){
    //         int out_index = k + b*l.c;

    //         float total = 0;
    //         for(i = 0; i < l.h*l.w; ++i){
    //             int in_index = i + l.h*l.w*(k + b*l.c);
    //             int out_index = in_index / 2;
    //             net.delta[in_index] += l.delta[out_index]/4;
    //         }
            
    //     }
    // }
}

void forward_multpool_layer2(const multpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    float avg = 0;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            // float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            // max_i = (val > max) ? index : max_i;
                            // max   = (val > max) ? val   : max;

                            float val_avg = (valid != 0) ? net.input[index] : 0;
                            avg += val_avg;
                        }
                    }
                    l.output[out_index] = avg/(l.size*l.size);
                    //l.output[out_index] = max;
                    //l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_multpool_layer2(const multpool_layer l, network net)
{
    int i,j,k,b;
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;

            float total = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                int out_index = in_index / 2;
                net.delta[in_index] += l.delta[out_index]/4;
            }
            
        }
    }
}