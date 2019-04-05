#ifndef MULTPOOL_LAYER_H
#define MULTPOOL_LAYER_H

#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer multpool_layer;

multpool_layer make_multpool_layer(int batch, int h, int w, int c, int size, int stride, int padding,int extra);
void resize_multpool_layer(multpool_layer *l, int w, int h);
void forward_multpool_layer1(const multpool_layer l, network net);
void backward_multpool_layer1(const multpool_layer l, network net);

void forward_multpool_layer2(const multpool_layer l, network net);
void backward_multpool_layer2(const multpool_layer l, network net);

#ifdef GPU
void forward_multpool_layer_gpu(multpool_layer l, network net);
void backward_multpool_layer_gpu(multpool_layer l, network net);
#endif

#endif

