//#define EIGEN_RUNTIME_NO_MALLOC
#include "internal.h"

double cnInvert(const CnMat *srcarr, CnMat *dstarr, enum cnInvertMethod method) {
	auto src = CONVERT_TO_EIGEN(srcarr);
	auto dst = CONVERT_TO_EIGEN(dstarr);

	EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(false);
	if (method == CN_INVERT_METHOD_LU) {
		dst.noalias() = src.inverse();
	} else {
		dst.noalias() = src.completeOrthogonalDecomposition().pseudoInverse();
	}
	return 0;
}


void cnSqRootSymmetric(const CnMat *srcarr, CnMat *dstarr) {
    auto src = CONVERT_TO_EIGEN(srcarr);
    auto dst = CONVERT_TO_EIGEN(dstarr);

    EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(false);
    dst.noalias() = Eigen::LLT<MatrixType>(src).matrixL().toDenseMatrix();
}

const int DECOMP_cnD = 1;
const int DECOMP_LU = 2;

extern "C" int cnSolve(const CnMat *_Aarr, const CnMat *_Barr, CnMat *_xarr, enum cnInvertMethod method) {
	auto Aarr = CONVERT_TO_EIGEN(_Aarr);
	auto Barr = CONVERT_TO_EIGEN(_Barr);
	auto xarr = CONVERT_TO_EIGEN(_xarr);

	if (method == CN_INVERT_METHOD_LU) {
		xarr.noalias() = Aarr.partialPivLu().solve(Barr);
	} else if (method == CN_INVERT_METHOD_QR) {
		xarr.noalias() = Aarr.colPivHouseholderQr().solve(Barr);
	} else {
		EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(true);
		auto cnd = Aarr.bdcSvd(
			Eigen::ComputeFullU |
			Eigen::ComputeFullV); // Eigen::JacobicnD<MatrixType>(Aarr, Eigen::ComputeFullU | Eigen::ComputeFullV);
		EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(false);
		xarr.noalias() = cnd.solve(Barr);
	}
	return 0;
}

void cnMulTransposed(const CnMat *src, CnMat *dst, int order, const CnMat *delta, double scale) {
	auto srcEigen = CONVERT_TO_EIGEN(src);
	auto dstEigen = CONVERT_TO_EIGEN(dst);

	if (delta) {
		auto deltaEigen = CONVERT_TO_EIGEN(delta);
		if (order == 0)
			dstEigen.noalias() = scale * (srcEigen - deltaEigen) * (srcEigen - deltaEigen).transpose();
		else
			dstEigen.noalias() = scale * (srcEigen - deltaEigen).transpose() * (src - delta);
	} else {
		if (order == 0)
			dstEigen.noalias() = scale * srcEigen * srcEigen.transpose();
		else
			dstEigen.noalias() = scale * srcEigen.transpose() * srcEigen;
	}
}

void cnTranspose(const CnMat *M, CnMat *dst) {
	auto src = CONVERT_TO_EIGEN(M);
	auto dstEigen = CONVERT_TO_EIGEN(dst);
	if (CN_FLT_PTR(M) == CN_FLT_PTR(dst))
		dstEigen = src.transpose().eval();
	else
		dstEigen.noalias() = src.transpose();
}

void print_mat(const CnMat *M);

double cnDet(const CnMat *M) {
	EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(false);
	auto MEigen = CONVERT_TO_EIGEN(M);
	return MEigen.determinant();
}
