#include "cnmatrix/cn_matrix.h"
#include <limits.h>
#include <string.h>

#ifdef _WIN32
#define CN_LOCAL_ONLY
#include <malloc.h>
#define alloca _alloca
#else
#define CN_LOCAL_ONLY __attribute__((visibility("hidden")))
#endif

#ifndef M_PI
# define M_PI		3.14159265358979323846
#endif

#define CN_Error(code, msg) assert(0 && msg); // cv::error( code, msg, CN_Func, __FILE__, __LINE__ )
CN_LOCAL_ONLY CnMat *cnCloneMat(const CnMat *mat) {
	CnMat *rtn = cnCreateMat(mat->rows, mat->cols);
	cnCopy(mat, rtn, 0);
	return rtn;
}

static size_t mat_size_bytes(const CnMat *mat) { return (size_t)sizeof(FLT) * mat->cols * mat->rows; }

FLT cnDistance(const CnMat *a, const CnMat *b) {
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);
    FLT r = 0;
    for (int i = 0; i < a->cols * a->rows; i++) {
        FLT dx = a->data[i] - b->data[i];
        r += dx*dx;
    }
    return FLT_SQRT(r);
}
FLT cnNorm2(const CnMat *s) {
  FLT r = 0;
  const FLT* in = s->data;
  for (int i = 0; i < s->cols * s->rows; i++) {
    r += in[i] * in[i];
  }
  return r;
}
FLT cnNorm(const CnMat *s) { return FLT_SQRT(cnNorm(s)); }
void cnSub(CnMat *dest, const CnMat *a, const CnMat *b) {
	assert(a->cols * a->rows == dest->cols * dest->rows);
	assert(b->cols * b->rows == dest->cols * dest->rows);
	for (int i = 0; i < (a->cols * a->rows); i++)
		dest->data[i] = a->data[i] - b->data[i];
}
void cnAdd(CnMat *dest, const CnMat *a, const CnMat *b) {
	assert(a->cols * a->rows == dest->cols * dest->rows);
	assert(b->cols * b->rows == dest->cols * dest->rows);
	for (int i = 0; i < (a->cols * a->rows); i++)
		dest->data[i] = a->data[i] + b->data[i];
}
void cnAddScaled(CnMat *dest, const CnMat *a, FLT as, const CnMat *b, FLT bs) {
	assert(a->cols * a->rows == dest->cols * dest->rows);
	assert(b->cols * b->rows == dest->cols * dest->rows);
	for (int i = 0; i < dest->cols * dest->rows; i++) {
		dest->data[i] = a->data[i] * as + b->data[i] * bs;
	}
}
void cnScale(CnMat *dest, const CnMat *a, FLT s) {
	assert(a->cols * a->rows == dest->cols * dest->rows);
	for (int i = 0; i < (a->cols * a->rows); i++)
	  dest->data[i] = a->data[i] * s;
}
void cnElementwiseMultiply(CnMat *dest, const CnMat *a, const CnMat *b) {
	assert(a->cols * a->rows == dest->cols * dest->rows);
	assert(b->cols * b->rows == dest->cols * dest->rows);
	for (int i = 0; i < dest->cols * dest->rows; i++) {
		dest->data[i] = a->data[i] * b->data[i];
	}
}

FLT cnDot(const CnMat* a, const CnMat* b) {
  assert(a->cols * a->rows == b->cols * b->rows);
  
  FLT rtn = 0;
  for (int i = 0; i < a->cols * a->rows; i++) {
    rtn += a->data[i] * b->data[i];
  }
  return rtn;
}

static inline FLT linmath_max(FLT x, FLT y) { return x > y ? x : y; }
static inline int linmath_imax(int x, int y) { return x > y ? x : y; }

static inline FLT linmath_min(FLT x, FLT y) { return x < y ? x : y; }
static inline int linmath_imin(int x, int y) { return x < y ? x : y; }

void cnCopy(const CnMat *src, CnMat *dest, const CnMat *mask) {
  assert(mask == 0 && "This isn't implemented yet");
  if (src->rows == dest->rows && src->cols == dest->cols) {
    memcpy(CN_RAW_PTR(dest), CN_RAW_PTR(src), mat_size_bytes(src));
  } else {
    for (int i = 0; i < linmath_imin(src->rows, dest->rows); i++)
      for (int j = 0; j < linmath_imin(src->cols, dest->cols); j++)
	cnMatrixSet(dest, i, j, cnMatrixGet(src, i, j));
  }       
}

FLT linmath_normrand(FLT mu, FLT sigma) {
    static const double epsilon = 0.0000001;

    static double z1= NAN;
    static bool generate = false;
    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);

    double z0 = sqrt(-2.0 * log(u1)) * cos(M_PI * 2. * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(M_PI * 2. * u2);
    return z0 * sigma + mu;
}

void cnRand(CnMat *arr, FLT mu, FLT sigma) {
    for (int i = 0; i < arr->rows; i++)
        for (int j = 0; j < arr->cols; j++)
            CN_RAW_PTR(arr)[i * arr->cols + j] = linmath_normrand(mu, sigma);
}

CN_LOCAL_ONLY void cnSetZero(CnMat *arr) {
	for (int i = 0; i < arr->rows; i++)
		for (int j = 0; j < arr->cols; j++)
			CN_RAW_PTR(arr)[i * arr->cols + j] = 0;
}
CN_LOCAL_ONLY void cvSetIdentity(CnMat *arr) {
	for (int i = 0; i < arr->rows; i++)
		for (int j = 0; j < arr->cols; j++)
			CN_RAW_PTR(arr)[i * arr->cols + j] = i == j;
}

CN_LOCAL_ONLY void cnReleaseMat(CnMat **mat) {
	free(*mat);
	*mat = 0;
}

/* the alignment of all the allocated buffers */
#define CN_MALLOC_ALIGN 16
CN_LOCAL_ONLY void *cnAlloc(size_t size) { return malloc(size); }
static inline void *cnAlignPtr(const void *ptr, int align) {
	assert((align & (align - 1)) == 0);
	return (void *)(((size_t)ptr + align - 1) & ~(size_t)(align - 1));
}

CN_LOCAL_ONLY void cnCreateData(CnMat *arr) {
  size_t step;
	CnMat *mat = (CnMat *)arr;
	step = mat->step;

	if (mat->rows == 0 || mat->cols == 0)
		return;

	if (CN_FLT_PTR(mat) != 0)
		CN_Error(CN_StsError, "Data is already allocated");

	if (step == 0)
		step = sizeof(FLT) * mat->cols;

	int64_t total_size = (int64_t)step * mat->rows;
	mat->data = (FLT *)cnAlloc(total_size);
}

CnMat *cnInitMatHeader(CnMat *arr, int rows, int cols) {
	assert(!(rows < 0 || cols < 0));

	int min_step = sizeof(FLT);
	assert(!(min_step <= 0));
	min_step *= cols;

	arr->step = min_step;
	arr->rows = rows;
	arr->cols = cols;
	arr->data = 0;

	return arr;
}

CN_LOCAL_ONLY CnMat *cnCreateMatHeader(int rows, int cols) {
	return cnInitMatHeader((CnMat *)cnAlloc(sizeof(CnMat)), rows, cols);
}
CN_LOCAL_ONLY CnMat *cnCreateMat(int height, int width) {
	CnMat *arr = cnCreateMatHeader(height, width);
	cnCreateData(arr);

	return arr;
}

inline void cn_ABAt_add(struct CnMat *out, const struct CnMat *A, const struct CnMat *B, const struct CnMat *C) {
	CN_CREATE_STACK_MAT(tmp, A->rows, B->cols);
	cnGEMM(A, B, 1, 0, 0, &tmp, 0);
	cnGEMM(&tmp, A, 1, C, 1, out, CN_GEMM_FLAG_B_T);
	CN_FREE_STACK_MAT(tmp);
}

void cnCopyROI(const CnMat *src, CnMat *dest, int start_i, int start_j) {
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            cnMatrixSet(dest, start_i + i, start_j + j, cnMatrixGet(src, i, j));
        }
    }
}