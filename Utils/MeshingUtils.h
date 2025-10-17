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

enum EnumMatIds {
  EDomain = 1,
  EBoundary = 2,
  ECylinder = 3,
  ECylinderBase = 4,
  EGoal = 5,
  ENone = -1
};

class MeshingUtils {

public:
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
                                     const TPZManVector<REAL, 3> &maxX);

  static TPZGeoMesh *ReadGeoMesh(std::string file);
};