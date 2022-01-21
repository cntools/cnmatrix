#define EIGEN_NO_DEBUG
//#pragma GCC optimize ("O3")

#include "cnmatrix/cn_matrix.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#ifdef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(v) EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(v)
#else
#define EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(v)
#endif

#ifdef cn_MATRIX_IS_COL_MAJOR
typedef Eigen::Matrix<FLT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 50, 50> MatrixType;
#else
typedef Eigen::Matrix<FLT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 50, 50> MatrixType;
#endif
typedef Eigen::OuterStride<> StrideType;
typedef Eigen::Map<MatrixType, 0, StrideType> MapType;

#define CONVERT_TO_EIGEN(A) MapType(A ? CN_FLT_PTR(A) : 0, A ? (A)->rows : 0, A ? (A)->cols : 0, StrideType(A ? (A)->step : 0))
