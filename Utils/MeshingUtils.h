#pragma once

#include <TPZCylinderMap.h>
#include <TPZGeoMeshTools.h>
#include <TPZGmshReader.h>
#include <TPZRefPatternDataBase.h>
#include <TPZRefPatternTools.h>
#include <TPZVTKGeoMesh.h>
#include <pzcheckgeom.h>
#include <pzgeoel.h>
#include <pzgeoelbc.h>
#include <pzvec_extras.h>
#include <tpzchangeel.h>
#include <tpzgeoelrefpattern.h>

class MeshingUtils {

public:

  struct BoundaryData
  {
    int matid = -1; // bc material ID
    int type;       // bc type 0: direct, 1: neumann
    REAL value;     // bc value
  };

  static REAL ElementDiameter(TPZGeoEl *gel);

  static TPZGeoMesh *CreateGeoMesh2D(const TPZManVector<int, 2> &nelDiv,
                                     const TPZManVector<REAL, 2> &minX,
                                     const TPZManVector<REAL, 2> &maxX);

    static TPZGeoMesh *CreateGeoMesh2D(const TPZManVector<int, 2> &nelDiv,
                                     const TPZManVector<REAL, 2> &minX,
                                     const TPZManVector<REAL, 2> &maxX,
                                     const TPZManVector<int, 5> &matIds);

  static TPZGeoMesh *CreateGeoMesh3D(const TPZManVector<int, 3> &nelDiv,
                                     const TPZManVector<REAL, 3> &minX,
                                     const TPZManVector<REAL, 3> &maxX,
                                     const TPZManVector<int, 7> &matIds);

  static TPZGeoMesh *ReadGeoMesh(std::string file, int EDomain, int EDirichlet, int EDirichlet2);
};