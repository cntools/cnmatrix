#include "internal.h"

extern "C" void cnSVD(CnMat *aarr, CnMat *warr, CnMat *uarr, CnMat *varr, enum cnSVDFlags flags) {
	auto aarrEigen = CONVERT_TO_EIGEN(aarr);
	auto warrEigen = CONVERT_TO_EIGEN(warr);

	int options = 0;
	if (uarr)
		options |= Eigen::ComputeFullU;
	if (varr)
		options |= Eigen::ComputeFullV;
	EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(true);
	auto cnd = aarrEigen.jacobiSvd(options);
	EIGEN_RUNTIME_SET_IS_MALLOC_ALLOWED(false);

	if (warrEigen.cols() == 1) {
		warrEigen.noalias() = cnd.singularValues();
	} else if (warrEigen.rows() == 1) {
		warrEigen.noalias() = cnd.singularValues().transpose();
	} else {
		warrEigen.diagonal().noalias() = cnd.singularValues();
	}

	if (uarr) {
		auto uarrEigen = CONVERT_TO_EIGEN(uarr);
		if (flags & CN_SVD_U_T)
			uarrEigen.noalias() = cnd.matrixU().transpose();
		else
			uarrEigen.noalias() = cnd.matrixU();
	}

	if (varr) {
		auto varrEigen = CONVERT_TO_EIGEN(varr);
		if (flags & CN_SVD_V_T)
			varrEigen.noalias() = cnd.matrixV().transpose();
		else
			varrEigen.noalias() = cnd.matrixV();
	}
}
