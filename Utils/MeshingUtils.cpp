#include "MeshingUtils.h"
#include <filesystem>
#include "TPZGenGrid2D.h"
#include "TPZGenGrid3D.h"

REAL MeshingUtils::ElementDiameter(TPZGeoEl *gel) {
  REAL maxdist = 0.;
  int nnodes = gel->NNodes();
  for (int i = 0; i < nnodes; ++i) {
    TPZManVector<REAL, 3> xi(3, 0.), xj(3, 0.);
    gel->Node(i).GetCoordinates(xi);
    for (int j = i + 1; j < nnodes; ++j) {
      gel->Node(j).GetCoordinates(xj);
      REAL dist = 0.;
      for (int d = 0; d < gel->Dimension(); ++d) {
        dist += (xi[d] - xj[d]) * (xi[d] - xj[d]);
      }
      dist = sqrt(dist);
      if (dist > maxdist)
        maxdist = dist;
    }
  }
  return maxdist;
}

TPZGeoMesh* MeshingUtils::ReadGeoMesh(std::string file) {

  std::string currentPath = std::filesystem::current_path();
  std::string fatherPath = std::filesystem::path(currentPath).parent_path();
  std::string path(fatherPath + "/" + file);
  TPZGeoMesh *gmesh = new TPZGeoMesh();
  {
    TPZGmshReader reader;
    TPZManVector<std::map<std::string, int>, 4> stringtoint(4);
    stringtoint[3]["volume_reservoir"] = EDomain;

    stringtoint[2]["surface_wellbore_cylinder"] = ECylinder;
    stringtoint[2]["surface_wellbore_heel"] = ECylinderBase;
    stringtoint[2]["surface_wellbore_toe"] = ECylinderBase;
    stringtoint[2]["surface_farfield"] = EFarfield;
    stringtoint[2]["surface_cap_rock"] = EFarfield;

    stringtoint[1]["curve_wellbore"] = ENone;
    stringtoint[1]["curve_heel"] = ENone;
    stringtoint[1]["curve_toe"] = ENone;

    stringtoint[0]["point_heel"] = ENone;
    stringtoint[0]["point_toe"] = ENone;

    reader.SetDimNamePhysical(stringtoint);
    reader.GeometricGmshMesh(path, gmesh);
  }

  //Remove gmsh boundary elements and create GeoElBC so normals are consistent
  int64_t nel = gmesh->NElements();
  for(int64_t el = 0; el < nel; el++){
      TPZGeoEl *gel = gmesh->Element(el);
      if(!gel || gel->Dimension() != gmesh->Dimension()-1) continue;
      TPZGeoElSide gelside(gel);
      TPZGeoElSide neigh = gelside.Neighbour();
      gel->RemoveConnectivities();
      int matid = gel->MaterialId();
      delete gel;
      TPZGeoElBC gbc(neigh, matid);
  }

  // Plot gmesh
  std::ofstream out("geomesh.vtk");
  TPZVTKGeoMesh::PrintGMeshVTK(gmesh, out);

  return gmesh;
}

TPZGeoMesh* MeshingUtils::CreateGeoMesh2D(const TPZManVector<int, 2> &nelDiv,
                            const TPZManVector<REAL, 2> &minX,
                            const TPZManVector<REAL, 2> &maxX) {

  TPZGeoMesh *gmesh = new TPZGeoMesh;
  TPZGenGrid2D generator(nelDiv, minX, maxX);
  generator.SetElementType(MMeshType::EQuadrilateral);
  generator.Read(gmesh, EDomain);
  generator.SetBC(gmesh, 4, EBoundary);
  generator.SetBC(gmesh, 5, EBoundary);
  generator.SetBC(gmesh, 6, EBoundary);
  generator.SetBC(gmesh, 7, EBoundary);

  return gmesh;
}

TPZGeoMesh* MeshingUtils::CreateGeoMesh3D(
  const TPZManVector<int, 3> &nelDiv, 
  const TPZManVector<REAL, 3> &minX, 
  const TPZManVector<REAL, 3> &maxX) {

  TPZGenGrid3D generator(minX, maxX, nelDiv, MMeshType::EHexahedral);

  generator.BuildVolumetricElements(EDomain);
  TPZGeoMesh *gmesh = generator.BuildBoundaryElements(EBoundary,
    EBoundary, EBoundary, EBoundary, EBoundary, EBoundary);

  return gmesh;
}