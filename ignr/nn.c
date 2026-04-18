#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N_SAMPLES 4
#define N_INPUT   2
#define N_HIDDEN  4
#define N_OUTPUT  1
#define EPOCHS    20000
#define LR        1.0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int n_in, n_out;
    double *W, *b, *z, *a;
} Layer;

typedef struct {
    Layer hidden;
    Layer output;
} Network;

/* ── Math ───────────────────────────────────────────── */
double randn() {
    double u1 = ((double)rand()+1) / ((double)RAND_MAX+1);
    double u2 = ((double)rand()+1) / ((double)RAND_MAX+1);
    return sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
}

double sigmoid(double x)       { return 1.0/(1.0+exp(-x)); }
double sigmoid_deriv(double a) { return a*(1.0-a); }  /* takes POST-activation */

void matmul(const double *A, const double *B, double *C, int ra, int ca, int cb) {
    for (int i=0;i<ra;i++) for (int j=0;j<cb;j++) {
        double s=0; for (int k=0;k<ca;k++) s+=A[i*ca+k]*B[k*cb+j];
        C[i*cb+j]=s;
    }
}

void transpose(const double *A, double *B, int r, int c) {
    for (int i=0;i<r;i++) for (int j=0;j<c;j++) B[j*r+i]=A[i*c+j];
}

void add_bias(double *M, const double *b, int rows, int cols) {
    for (int i=0;i<rows;i++) for (int j=0;j<cols;j++) M[i*cols+j]+=b[j];
}

/* ── Memory ─────────────────────────────────────────── */
Layer create_layer(int n_in, int n_out) {
    Layer l = {n_in, n_out,
        malloc(n_in*n_out*sizeof(double)),
        malloc(n_out*sizeof(double)),
        malloc(N_SAMPLES*n_out*sizeof(double)),
        malloc(N_SAMPLES*n_out*sizeof(double))};

    double scale = sqrt(2.0 / n_in);   /* FIX 1: was sqrt(2.0, n_in) */
    for (int i=0;i<n_in*n_out;i++) l.W[i] = randn() * scale;
    for (int i=0;i<n_out;i++)      l.b[i] = 0.0;
    return l;
}

void free_layer(Layer *l) {
    free(l->W); free(l->b); free(l->z); free(l->a);
    l->W = l->b = l->z = l->a = NULL;
}

Network create_network() {
    Network net;
    net.hidden = create_layer(N_INPUT,  N_HIDDEN);
    net.output = create_layer(N_HIDDEN, N_OUTPUT);
    return net;
}

void free_network(Network *net) {
    free_layer(&net->hidden);
    free_layer(&net->output);
}

/* ── Forward ────────────────────────────────────────── */
void forward(Network *net, const double *X) {
    matmul(X, net->hidden.W, net->hidden.z, N_SAMPLES, N_INPUT, N_HIDDEN);
    add_bias(net->hidden.z, net->hidden.b, N_SAMPLES, N_HIDDEN);
    for (int i=0;i<N_SAMPLES*N_HIDDEN;i++)
        net->hidden.a[i] = sigmoid(net->hidden.z[i]);

    matmul(net->hidden.a, net->output.W, net->output.z, N_SAMPLES, N_HIDDEN, N_OUTPUT);
    add_bias(net->output.z, net->output.b, N_SAMPLES, N_OUTPUT);
    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++)
        net->output.a[i] = sigmoid(net->output.z[i]);
}

/* ── Loss ───────────────────────────────────────────── */
double mse_loss(const double *pred, const double *Y, int n) {
    double s=0; for (int i=0;i<n;i++) { double d=pred[i]-Y[i]; s+=d*d; } return s/n;
}

/* ── Backward ───────────────────────────────────────── */
void backward(Network *net, const double *X, const double *Y) {
    double dl_da2[N_SAMPLES*N_OUTPUT], dl_dz2[N_SAMPLES*N_OUTPUT];
    double dl_dW2[N_HIDDEN*N_OUTPUT],  dl_db2[N_OUTPUT];
    double dl_da1[N_SAMPLES*N_HIDDEN], dl_dz1[N_SAMPLES*N_HIDDEN];
    double dl_dW1[N_INPUT*N_HIDDEN],   dl_db1[N_HIDDEN];

    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++)
        dl_da2[i] = 2.0*(net->output.a[i]-Y[i])/N_SAMPLES;
    for (int i=0;i<N_SAMPLES*N_OUTPUT;i++)
        dl_dz2[i] = dl_da2[i] * sigmoid_deriv(net->output.a[i]);

    { double a1T[N_HIDDEN*N_SAMPLES];
      transpose(net->hidden.a, a1T, N_SAMPLES, N_HIDDEN);
      matmul(a1T, dl_dz2, dl_dW2, N_HIDDEN, N_SAMPLES, N_OUTPUT); }
    for (int j=0;j<N_OUTPUT;j++) {
        double s=0; for (int i=0;i<N_SAMPLES;i++) s+=dl_dz2[i*N_OUTPUT+j];
        dl_db2[j]=s; }

    { double W2T[N_OUTPUT*N_HIDDEN];
      transpose(net->output.W, W2T, N_HIDDEN, N_OUTPUT);
      matmul(dl_dz2, W2T, dl_da1, N_SAMPLES, N_OUTPUT, N_HIDDEN); }
    for (int i=0;i<N_SAMPLES*N_HIDDEN;i++)
        dl_dz1[i] = dl_da1[i] * sigmoid_deriv(net->hidden.a[i]);

    { double XT[N_INPUT*N_SAMPLES];
      transpose(X, XT, N_SAMPLES, N_INPUT);
      matmul(XT, dl_dz1, dl_dW1, N_INPUT, N_SAMPLES, N_HIDDEN); }
    for (int j=0;j<N_HIDDEN;j++) {
        double s=0; for (int i=0;i<N_SAMPLES;i++) s+=dl_dz1[i*N_HIDDEN+j];
        dl_db1[j]=s; }

    for (int i=0;i<N_HIDDEN*N_OUTPUT;i++) net->output.W[i] -= LR*dl_dW2[i];
    for (int i=0;i<N_OUTPUT;i++)          net->output.b[i] -= LR*dl_db2[i];
    for (int i=0;i<N_INPUT*N_HIDDEN;i++)  net->hidden.W[i] -= LR*dl_dW1[i];
    for (int i=0;i<N_HIDDEN;i++)          net->hidden.b[i] -= LR*dl_db1[i];
}

/* ── Main ───────────────────────────────────────────── */
int main(void) {
    srand(time(NULL));

    double X[N_SAMPLES*N_INPUT]  = {0,0, 0,1, 1,0, 1,1};
    double Y[N_SAMPLES*N_OUTPUT] = {0,1,1,0};

    Network net = create_network();

    for (int epoch=0; epoch<=EPOCHS; epoch++) {
        forward(&net, X);
        double loss = mse_loss(net.output.a, Y, N_SAMPLES);
        if (epoch % 2000 == 0) printf("Epoch %5d | Loss: %.4f\n", epoch, loss);
        if (epoch < EPOCHS) backward(&net, X, Y);
    }

    printf("\nFinal predictions:\n");
    double ins[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double tgts[4]   = {0,1,1,0};
    for (int i=0;i<N_SAMPLES;i++)
        printf("  [%.0f %.0f] -> %.3f  (target: %.0f)\n",
               ins[i][0], ins[i][1], net.output.a[i], tgts[i]);

    free_network(&net);
    return 0;
}