#include "include/darknet.h"
#include <stdio.h>

extern image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);

void test_image(char *filename)
{
    image im = load_image(filename, 0,0, 3);
    float mag = mag_array(im.data, im.w*im.h*im.c);
    printf("L2 Norm: %f\n", mag);
    image gray = grayscale_image(im);

    image c1 = copy_image(im);
    image c2 = copy_image(im);
    image c3 = copy_image(im);
    image c4 = copy_image(im);
    //distort_image(c1, .1, 1.5, 1.5);
    c1 = resize_image(c1,256,256);

    //c1 = rotate_crop_image(c1,0.2,1.,773,512,200,0.,1.);

    distort_image(c2, -.1, .66666, .66666);
    distort_image(c3, .1, 1.5, .66666);
    distort_image(c4, .1, .66666, 1.5);

    


    show_image(im,   "Original", 1);
    show_image(gray, "Gray", 1);
    show_image(c1, "C1", 1);
    show_image(c2, "C2", 1);
    show_image(c3, "C3", 1);
    show_image(c4, "C4", 1);
}

data load_humanseg_data(){
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;

    return d;
}


void test_predict(){
    const char* cfg = "cfg/humanseg.cfg";
    const char* weights = "darknet.weights";

    network *net = parse_network_cfg(cfg);
    load_weights_upto(net, weights,0,6);

    const char *test = "data/humanseg/test.png";
    image im = load_image(test, 256,256, 3);
    float *X = im.data;
    //set_batch_network(net,1);
    network_predict(net, X);
    image pre = get_network_image(net);
    show_image(pre, "data/humanseg/test_pred", 1);
}

void test_train(){
    const char* cfg = "cfg/humanseg.cfg";
    const char* weights = "darknet.weights";
    char *train_images = "data/humanseg/train.list";
    char *train_labels = "data/humanseg/label.list";
    char *backup_directory = "data/humanseg/backup/";
    srand(time(0));
    float avg_loss = -1;
    network *net = parse_network_cfg(cfg);
    load_weights_upto(net, weights,0,6);

    *net->seen = 0;

    char *base = basecfg(cfg);

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    int i = *net->seen/imgs;
    data train, buffer;

    list *plist = get_paths(train_images);
    list *llist = get_paths(train_labels);
    //assert(plist->size == llist->size);
    char **paths = (char **)list_to_array(plist);
    char **labels = (char **)list_to_array(llist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.labels = labels;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = HUMANSEG_DATA;

    args.min = net->min_ratio*args.w;
    args.max = net->max_ratio*args.w;

    args.angle = net->angle;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    
    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    //printf("%d\n", *net->seen);

    while(get_current_batch(net) < net->max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);


        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            const char *test = "data/humanseg/test.png";
            image im = load_image(test, 256,256, 3);
            float *X = im.data;
            int batch = net->batch;
            set_batch_network(net,1);
            network_predict(net, X);
            image pre = get_network_image(net);
            show_image(pre, "data/humanseg/test_pred", 1);
            set_batch_network(net,batch);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_network(const char *name){
    network *net = parse_network_cfg(name);
}

int main(int argc, char **argv){
    test_train();
    //print_network("cfg/humanseg.cfg");
    return 0;
}