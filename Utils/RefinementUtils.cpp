#include "RefinementUtils.h"
#include "MeshingUtils.h"
#include "pzcheckgeom.h"

void RefinementUtils::MeshSmoothing(TPZGeoMesh* gmesh, TPZVec<int>& refinementIndicator) {
  // If an element has most of its neighbors refined, then refine it too
  int ngel = gmesh->NElements();
  for (int64_t iel = 0; iel < refinementIndicator.size(); ++iel) {
    if (refinementIndicator[iel] != 1) continue;
    TPZGeoEl* gel = gmesh->Element(iel);
    if (!gel) DebugStop();
    int firstside = gel->FirstSide(gel->Dimension()-1);
    int lastside = gel->FirstSide(gel->Dimension());
    for (int side = firstside; side < lastside; ++side) {
      TPZGeoElSide gelSide(gel, side);
      TPZGeoElSide neigh = gelSide.Neighbour();
      TPZGeoEl* neighGel = neigh.Element();
      if (!neighGel) DebugStop();
      int threshold = 4;
      int neighIndex = neighGel->Index();
      if (refinementIndicator[neighIndex] == 1) continue;
      int firstsideNeigh = neighGel->FirstSide(neighGel->Dimension()-1);
      int lastsideNeigh = neighGel->FirstSide(neighGel->Dimension());
      int countRefNeigh = 0;
      for (int sideNeigh = firstsideNeigh; sideNeigh < lastsideNeigh; ++sideNeigh) {
        TPZGeoElSide neighSide(neighGel, sideNeigh);
        TPZGeoElSide neigh2 = neighSide.Neighbour();
        if (!neigh2) DebugStop();
        int64_t neigh2Index = neigh2.Element()->Index();
        if (refinementIndicator[neigh2Index] == 1) countRefNeigh++;
      }
      if (countRefNeigh >= threshold) {
        refinementIndicator[neighGel->Index()] = 1;
      }
    }  
  }

  // If an element has a neighbor with refinement two levels higher, then refine it too
  // for (int64_t iel = 0; iel < refinementIndicator.size(); ++iel) {
  //   if (refinementIndicator[iel] != 1) continue;
  //   TPZGeoEl* gel = gmesh->Element(iel);
  //   if (!gel) DebugStop();
  //   int matid = gel->MaterialId();
  //   int firstside = gel->FirstSide(gel->Dimension()-1);
  //   int lastside = gel->FirstSide(gel->Dimension());
  //   for (int side = firstside; side < lastside; ++side) {
  //     TPZGeoElSide gelSide(gel, side);
  //     TPZGeoElSide lowerSide = gelSide.HasLowerLevelNeighbour(matid);
  //     if (lowerSide) {
  //       TPZGeoEl* neighGel = lowerSide.Element();
  //       if (!neighGel->HasSubElement()) {
  //         refinementIndicator[neighGel->Index()] = 1;
  //         DebugStop();
  //       }
  //     }
  //   }
  // }
  for (int64_t iel = 0; iel < refinementIndicator.size(); ++iel) {
    if (refinementIndicator[iel] != 1) continue;
    TPZGeoEl* gel = gmesh->Element(iel);

    // Get corner idexes of gel
    TPZManVector<int64_t> cornerIndexes(gel->NCornerNodes());
    for (int i = 0; i < gel->NCornerNodes(); ++i) {
      cornerIndexes[i] = gel->NodeIndex(i);
    }
    TPZGeoEl* fatherGel = gel->Father();
    if (!fatherGel) continue;
    int firstside = fatherGel->FirstSide(fatherGel->Dimension()-1);
    int lastside = fatherGel->FirstSide(fatherGel->Dimension());
    for (int side = firstside; side < lastside; ++side) {
      TPZGeoElSide gelSide(fatherGel, side);
      TPZGeoElSide neigh = gelSide.Neighbour();
      TPZGeoEl* neighGel = neigh.Element();
      if (!neighGel) DebugStop();
      if (neighGel->Dimension() != gel->Dimension()) continue;
      if (neighGel->HasSubElement()) continue;

      // Get corner idexes of neighGel
      TPZManVector<int64_t> neighCornerIndexes(neighGel->NCornerNodes());
      for (int i = 0; i < neighGel->NCornerNodes(); ++i) {
        neighCornerIndexes[i] = neighGel->NodeIndex(i);
      }

      // Verify if there are common corners
      int commonCorners = 0;
      for (int i = 0; i < gel->NCornerNodes(); ++i) {
        for (int j = 0; j < neighGel->NCornerNodes(); ++j) {
          if (cornerIndexes[i] == neighCornerIndexes[j]) {
            commonCorners++;
            break;
          }
        }
      }
      
      if (commonCorners > 0) refinementIndicator[neighGel->Index()] = 1;
    }
  }
}

void RefinementUtils::UniformRefinement(TPZGeoMesh *gmesh) {
  TPZCheckGeom checkgeom(gmesh);
  checkgeom.UniformRefine(1);
}

void RefinementUtils::AdaptiveRefinement(TPZGeoMesh *gmesh, TPZVec<int>& refinementIndicator) {
  int dim = gmesh->Dimension();

  for (int64_t iel = 0; iel < refinementIndicator.size(); ++iel) {
    if (refinementIndicator[iel] == 0) continue;
    TPZGeoEl *gel = gmesh->Element(iel);
    if (!gel) continue;
    if (gel->HasSubElement()) continue;
    TPZVec<TPZGeoEl *> pv;
    gel->Divide(pv);

    // Refine boundary
    int firstside = gel->FirstSide(dim-1);
    int lastside = gel->FirstSide(dim);
    for (int side = firstside; side < lastside; ++side) {
      TPZGeoElSide gelSide(gel, side);
      std::set<int> bcIds = {EBoundary, EGoal, ECylinder, ECylinderBase};
      TPZGeoElSide neigh = gelSide.HasNeighbour(bcIds);
      if (neigh) {
        TPZGeoEl *neighGel = neigh.Element();
        if (neighGel->Dimension() != dim-1) continue;
        TPZVec<TPZGeoEl *> pv2;
        neighGel->Divide(pv2);
      }
    }
  }
}