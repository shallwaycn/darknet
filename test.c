#include "include/darknet.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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

void calculate_loss(float *output, float *delta, int n, float thresh)
{
    int i;
    float mean = mean_array(output, n); 
    float var = variance_array(output, n);
    for(i = 0; i < n; ++i){
        if(delta[i] > mean + thresh*sqrt(var)) delta[i] = output[i];
        else delta[i] = 0;
    }
}

void test_predict(){
    const char* cfg = "cfg/humanseg3.cfg";
    const char* weights = "data/humanseg/backup/humanseg3.backup";

    const char* img = "data/humanseg/sized.jpg";
    const char* img_m = "data/humanseg/sized_m.jpg";

    network *net = parse_network_cfg(cfg);
    load_weights(net, weights);
    set_batch_network(net,1);

    image im = load_image(img, 256,256, 3);
    image im_m = load_image(img_m, 256,256, 3);

    //guide network here
    //layer last = net->layers[net->n-1];
    //copy_cpu(net->inputs, im.data, 1, net->input, 1);
    // forward_network(net);
    // copy_cpu(last.outputs, last.output, 1, last.delta, 1);
    // calculate_loss(last.output, last.delta, last.outputs, 1);
    // backward_network(net);

    //guide network here
    copy_cpu(net->inputs, im.data, 1, net->input, 1);
    copy_cpu(net->outputs, im_m.data, 1, net->truth, 1);
    net->learning_rate = 0.01;

    for (int i = 0; i < 2; i++){
        train_network_datum(net);
    }
    
    
    float *X = im.data;
    network_predict(net, X);
    image pre = get_network_image(net);
    show_image(pre, "data/humanseg/test_pred", 1);
}

void test_train(){
    
    //const char* weights = "resnext50.weights";
    const char* weights = "darknet.weights";
    const char* cfg = "cfg/humanseg6.cfg";
    
    char *train_images = "data/humanseg/train.list";
    char *train_labels = "data/humanseg/label.list";
    char *backup_directory = "data/humanseg/backup/";
    char *val_directory = "data/humanseg/val_output/";

    char *val_images = "data/humanseg/val.list";
    srand(time(0));
    float avg_loss = -1;
    network *net = parse_network_cfg(cfg);
    load_weights_upto(net, weights,0,2);

    *net->seen = 0;

    char *base = basecfg(cfg);

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    int i = *net->seen/imgs;
    data train, buffer;

    list *plist = get_paths(train_images);
    list *llist = get_paths(train_labels);
    list *vlist = get_paths(val_images);
    //assert(plist->size == llist->size);
    char **paths = (char **)list_to_array(plist);
    char **labels = (char **)list_to_array(llist);
    char **vals = (char **)list_to_array(vlist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.labels = labels;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = HUMANSEG_DATA;

    args.min = net->min_crop;
    args.max = net->max_crop;

    args.aspect = net->aspect;

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
        float loss = train_network_sgd(net, train,net->subdivisions);
        //sleep(1);
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

            int batch = net->batch;
            set_batch_network(net,1);

            struct stat st = {0};

            sprintf(buff, "%s/%d/", val_directory,i);
            if (stat(buff, &st) == -1) {
                mkdir(buff, 0700);
            }

            for (int j = 0; j < vlist->size; j++){
                image im = load_image(vals[j], 256,256, 3);
                float *X = im.data;
                
                network_predict(net, X);
                image pre = get_network_image(net);

                sprintf(buff, "%s/%d/%s", val_directory,i,basecfg(vals[j]));

                show_image(pre, buff, 1);

                free_image(im);
                //free_image(pre);
            }

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
    //test_predict();
    //print_network("cfg/humanseg.cfg");
    return 0;
}