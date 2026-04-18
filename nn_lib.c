/*
 * nn_lib.c — Neural network as a shared library
 * Python calls train() and predict() via ctypes.
 *
 * Compile (Linux/Mac):
 *   gcc -shared -fPIC -o nn_lib.so nn_lib.c -lm
 *
 * Compile (Windows MinGW):
 *   gcc -shared -o nn_lib.dll nn_lib.c -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//Fixed for XOR network
#define N_SAMPLES 4
#define N_INPUT   2
#define N_HIDDEN  4
#define N_OUTPUT  1

typedef struct {
    int n_in, n_out;
    double *W, *b, *z, *a;
} Layer;

typedef struct {
    Layer hidden;
    Layer output;
} Network;

static Network net;
static int net_ready = 0;

//maths defining:
static double randn() {
    double u1 = ((double)rand()+1) / ((double)RAND_MAX+1);
    double u2 = ((double)rand()+1) / ((double)RAND_MAX+1);
    return sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
}

static double sigmoid(double x)       { return 1.0/(1.0+exp(-x)); }
static double sigmoid_deriv(double a) { return a*(1.0-a); }

static void matmul(const double *A, const double *B, double *C,
                   int ra, int ca, int cb) {
    for (int i=0;i<ra;i++) for (int j=0;j<cb;j++) {
        double s=0; for (int k=0;k<ca;k++) s+=A[i*ca+k]*B[k*cb+j];
        C[i*cb+j]=s;
    }
}

static void transpose(const double *A, double *B, int r, int c) {
    for (int i=0;i<r;i++) for (int j=0;j<c;j++) B[j*r+i]=A[i*c+j];
}

static void add_bias(double *M, const double *b, int rows, int cols) {
    for (int i=0;i<rows;i++) for (int j=0;j<cols;j++) M[i*cols+j]+=b[j];
}

//Memory allocation function:
static Layer create_layer(int n_in, int n_out) {
    Layer l = {n_in, n_out,
        malloc(n_in*n_out*sizeof(double)),
        malloc(n_out*sizeof(double)),
        malloc(N_SAMPLES*n_out*sizeof(double)),
        malloc(N_SAMPLES*n_out*sizeof(double))};
    double scale = sqrt(2.0 / n_in);
    for (int i=0;i<n_in*n_out;i++) l.W[i] = randn()*scale;
    for (int i=0;i<n_out;i++)      l.b[i] = 0.0;
    return l;
}

static void free_layer(Layer *l) {
    free(l->W); free(l->b); free(l->z); free(l->a);
    l->W = l->b = l->z = l->a = NULL;
}

//Forward passing function:
static void forward(const double *X) {
    matmul(X, net.hidden.W, net.hidden.z, N_SAMPLES, N_INPUT, N_HIDDEN);
    add_bias(net.hidden.z, net.hidden.b, N_SAMPLES, N_HIDDEN);
    for (int i=0;i<N_SAMPLES*N_HIDDEN;i++)
        net.hidden.a[i] = sigmoid(net.hidden.z[i]);

    matmul(net.hidden.a, net.output.W, net.output.z, N_SAMPLES, N_HIDDEN, N_OUTPUT);
    add_bias(net.output.z, net.output.b, N_SAMPLES, N_OUTPUT);
    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++)
        net.output.a[i] = sigmoid(net.output.z[i]);
}
//MSE Loss calculation:
static double mse_loss(const double *Y) {
    double s=0;
    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++) {
        double d = net.output.a[i]-Y[i]; s+=d*d;
    }
    return s / (N_SAMPLES*N_OUTPUT);
}
//Backpropagation function:
static void backward(const double *X, const double *Y, double lr) {
    double dl_da2[N_SAMPLES*N_OUTPUT], dl_dz2[N_SAMPLES*N_OUTPUT];
    double dl_dW2[N_HIDDEN*N_OUTPUT],  dl_db2[N_OUTPUT];
    double dl_da1[N_SAMPLES*N_HIDDEN], dl_dz1[N_SAMPLES*N_HIDDEN];
    double dl_dW1[N_INPUT*N_HIDDEN],   dl_db1[N_HIDDEN];

    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++)
        dl_da2[i] = 2.0*(net.output.a[i]-Y[i])/N_SAMPLES;
    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++)
        dl_dz2[i] = dl_da2[i]*sigmoid_deriv(net.output.a[i]);

    { double a1T[N_HIDDEN*N_SAMPLES];
      transpose(net.hidden.a,a1T,N_SAMPLES,N_HIDDEN);
      matmul(a1T,dl_dz2,dl_dW2,N_HIDDEN,N_SAMPLES,N_OUTPUT); }
    for (int j=0;j<N_OUTPUT;j++) {
        double s=0; for(int i=0;i<N_SAMPLES;i++) s+=dl_dz2[i*N_OUTPUT+j];
        dl_db2[j]=s; }

    { double W2T[N_OUTPUT*N_HIDDEN];
      transpose(net.output.W,W2T,N_HIDDEN,N_OUTPUT);
      matmul(dl_dz2,W2T,dl_da1,N_SAMPLES,N_OUTPUT,N_HIDDEN); }
    for (int i=0;i<N_SAMPLES*N_HIDDEN;i++)
        dl_dz1[i] = dl_da1[i]*sigmoid_deriv(net.hidden.a[i]);

    { double XT[N_INPUT*N_SAMPLES];
      transpose(X,XT,N_SAMPLES,N_INPUT);
      matmul(XT,dl_dz1,dl_dW1,N_INPUT,N_SAMPLES,N_HIDDEN); }
    for (int j=0;j<N_HIDDEN;j++) {
        double s=0; for(int i=0;i<N_SAMPLES;i++) s+=dl_dz1[i*N_HIDDEN+j];
        dl_db1[j]=s; }

    for (int i=0;i<N_HIDDEN*N_OUTPUT;i++) net.output.W[i] -= lr*dl_dW2[i];
    for (int i=0;i<N_OUTPUT;i++)          net.output.b[i] -= lr*dl_db2[i];
    for (int i=0;i<N_INPUT*N_HIDDEN;i++)  net.hidden.W[i] -= lr*dl_dW1[i];
    for (int i=0;i<N_HIDDEN;i++)          net.hidden.b[i] -= lr*dl_db1[i];
}

//PUBLIC API

//init_network()
void init_network(void) {
    if (net_ready) {
        free_layer(&net.hidden);
        free_layer(&net.output);
    }
    srand(time(NULL));
    net.hidden = create_layer(N_INPUT,  N_HIDDEN);
    net.output = create_layer(N_HIDDEN, N_OUTPUT);
    net_ready = 1;
}

//train(X, Y, epochs, lr, loss_out)
void train(const double *X, const double *Y,
           int epochs, double lr, double *loss_out) {
    for (int e=0; e<epochs; e++) {
        forward(X);
        loss_out[e] = mse_loss(Y);
        backward(X, Y, lr);
    }
}

//predict(X, out)
void predict(const double *X, double *out) {
    forward(X);
    for (int i=0; i<N_SAMPLES*N_OUTPUT; i++)
        out[i] = net.output.a[i];
}

//Free all memory
void destroy_network(void) {
    if (net_ready) {
        free_layer(&net.hidden);
        free_layer(&net.output);
        net_ready = 0;
    }
}