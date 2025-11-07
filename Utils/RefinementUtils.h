#pragma once

#include <TPZGeoMeshTools.h>
#include <TPZVTKGeoMesh.h>
#include <pzcheckgeom.h>
#include <tpzgeoelrefpattern.h>

class RefinementUtils {

public:
  static void UniformRefinement(TPZGeoMesh *gmesh);

  static void MeshSmoothing(TPZGeoMesh *gmesh, TPZVec<int> &refinementIndicator);

  static void AdaptiveRefinement(TPZGeoMesh *gmesh, TPZVec<int> &refinementIndicator);

  static void ConvergenceOrder(TPZFMatrix<REAL> &errors, TPZVec<REAL> &hs);
};