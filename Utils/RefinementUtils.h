#pragma once

#include <TPZGeoMeshTools.h>
#include <TPZVTKGeoMesh.h>
#include <pzcheckgeom.h>
#include <tpzgeoelrefpattern.h>

// Internal stuff
#include "MeshingUtils.h"

class RefinementUtils {

public:
  static void UniformRefinement(TPZGeoMesh *gmesh);

  static void MarkToRefine(TPZVec<REAL> &elementErrors, TPZVec<int> &refinementIndicator, REAL threshold);

  static void MarkToRefineNew(TPZVec<REAL> &elementErrors, TPZVec<int> &refinementIndicator, REAL threshold);

  static void MeshSmoothing(TPZGeoMesh *gmesh, TPZVec<int> &refinementIndicator);

  static void RefineBoundary(TPZGeoMesh* gmesh, TPZVec<int>& refinementIndicator, TPZVec<MeshingUtils::BoundaryData>& bcData);

  static void AdaptiveRefinement(TPZGeoMesh *gmesh, TPZVec<int> &refinementIndicator);

  static void ConvergenceOrder(TPZFMatrix<REAL> &errors, TPZVec<REAL> &hs);
};