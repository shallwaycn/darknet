#include "attentionrefine_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

layer make_attentionrefine_layer(int batch, int index, int w2, int h2, int c2,int w, int h, int c)
{
    fprintf(stderr, "are  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {0};
    l.type = ATTENTIONREFINE;
    assert(w2 == 1);
    assert(h2 == 1);
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = c;

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));

    l.forward = forward_attentionrefine_layer;
    l.backward = backward_attentionrefine_layer;
    #ifdef GPU
    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void resize_attentionrefine_layer(layer *l, int w, int h)
{
    l->w = 1;
    l->h = 1;

    l->out_w = w;
    l->out_h = h;

    l->outputs = w*h*l->out_c;
    l->inputs = 1 * 1 * l->out_c;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}
void multiply_cpu_forward(int batch, int w1, int h1, int c1, float *mult, int w2, int h2, int c2,float *out)
{
    float sample = w1/w2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c2; ++k){
            for(j = 0; j < h2; ++j){
                for(i = 0; i < w2; ++i){
                    int out_index = i + w2*(j + h2*(k + c2*b));
                    int mult_index = i*0 + w1*(j*0 + h1*(k + c2*b));
                    out[out_index] = out[out_index] * mult[mult_index];
                }
            }
        }
    }
}

void forward_attentionrefine_layer(const layer l, network net)
{
    //printf("%f:%f\n",net.input[0],net.truth[0]);
    copy_cpu(l.outputs*l.batch, net.layers[l.index].output, 1, l.output, 1);
    multiply_cpu_forward(l.batch, l.w, l.h, l.c, net.input, l.out_w, l.out_h, l.out_c,l.output);
    //activate_array(l.output, l.outputs*l.batch, LINEAR);
}

void backward_attentionrefine_layer(const layer l, network net)
{
    //gradient_array(l.output, l.outputs*l.batch, LINEAR, l.delta);
    int i,j,k,b;
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;

            float total = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                total += l.delta[in_index];
                //net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
            //int in_index = 0 + l.w*(0 + l.h*(k + l.c*b));
            //printf("%d:%f\n",in_index,l.delta[in_index]);
            net.delta[out_index] = total / l.h*l.w;
        }
    }
    //multiply_cpu_backward(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c,net.delta);
    //axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
    //multiply_cpu_backward(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
