#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <omp.h>

//#define N 25000
#define N 1500
#define t 0.00001f
#define eps 0.00001f

void fill_vector(double *vector, const int length, const double fill_value) {
#pragma omp parallel for
    for (size_t i = 0; i < length; i++) {
        vector[i] = fill_value;
    }
}

//TODO: second for
void fill_matrix(double *A, const int height, const int width) {
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == j)
                A[i * width + j] = 2.0f;
            else
                A[i * width + j] = 1.0f;
        }
    }
}

//TODO: second for
void mult_matr_on_vect(const double *A, const int height, const int width, const double *vect, const int vect_len,
                       double *res) {
    if (width != vect_len) {
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        double summ = 0;
        for (int j = 0; j < width; j++) {
            summ += A[i * width + j] * vect[j];
        }
        res[i] = summ;
    }
}

void diff_vector(const double *vect_1, const int len_1, const double *vect_2, const int len_2, double *res) {
    if (len_1 != len_2) {
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < len_1; i++) {
        res[i] = vect_1[i] - vect_2[i];
    }

}

void mult_vect_on_num(const double *vect, const int vect_len, const double number, double *res) {
#pragma omp parallel for
    for (int i = 0; i < vect_len; i++) {
        res[i] = vect[i] * number;
    }
}

void make_copy(const double *vect_1, const int len_1, double *vect_2, const int len_2) {
    if (len_1 != len_2) {
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < len_1; i++) {
        vect_2[i] = vect_1[i];
    }
}

double norm(const double *vect, const int vect_len) {
    double summ = 0;
#pragma omp parallel for reduction(+:summ)
    for (int i = 0; i < vect_len; i++) {
        summ += vect[i] * vect[i];
    }
    return summ;
}

int check(double vect_norm, double b_norm) {
    if (vect_norm / b_norm < eps * eps) {
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {

    double *A = calloc(N * N, sizeof(double));
    double *b = calloc(N, sizeof(double));
    double *x_prev = calloc(N, sizeof(double));
    double *x_next = calloc(N, sizeof(double));
    int flag = 1;
    double tmp_norm = 0;

    double start_time = omp_get_wtime();

    fill_matrix(A, N, N);
    fill_vector(b, N, (double) (N + 1));
    double b_norm = norm(b, N);
    fill_vector(x_prev, N, 0);

    while (flag) {
        mult_matr_on_vect(A, N, N, x_prev, N, x_next);
        diff_vector(x_next, N, b, N, x_next);
        tmp_norm = norm(x_next, N);
        flag = check(tmp_norm, b_norm);
        mult_vect_on_num(x_next, N, t, x_next);
        diff_vector(x_prev, N, x_next, N, x_next);
        make_copy(x_next, N, x_prev, N);
    }

    double end_time = omp_get_wtime();

    printf("Result vector element: %f\n", x_prev[0]);
    printf("Last calculated (Ax-b) norm: %f\n", tmp_norm);
    printf("Elapsed time: %f sec\n", (end_time - start_time));

    return 0;
}