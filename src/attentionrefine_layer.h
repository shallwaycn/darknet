#ifndef ATTENTIONREFINE_LAYER_H
#define ATTENTIONREFINE_LAYER_H

#include "layer.h"
#include "network.h"

layer make_attentionrefine_layer(int batch, int index, int w2, int h2, int c2, int w, int h, int c);
void forward_attentionrefine_layer(const layer l, network net);
void backward_attentionrefine_layer(const layer l, network net);
void resize_attentionrefine_layer(layer *l, int w, int h);

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
#endif

#endif
