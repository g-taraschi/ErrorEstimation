#include <filesystem>
#include "MeshingUtils.h"
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

TPZGeoMesh* MeshingUtils::CreateGeoMesh2D(const TPZManVector<int, 2> &nelDiv,
                            const TPZManVector<REAL, 2> &minX,
                            const TPZManVector<REAL, 2> &maxX,
                            const TPZManVector<int, 5> &matIds) {

  TPZGeoMesh *gmesh = new TPZGeoMesh;
  TPZGenGrid2D generator(nelDiv, minX, maxX);
  generator.SetElementType(MMeshType::EQuadrilateral);
  generator.Read(gmesh, matIds[0]);
  generator.SetBC(gmesh, 4, matIds[1]);
  generator.SetBC(gmesh, 5, matIds[2]);
  generator.SetBC(gmesh, 6, matIds[3]);
  generator.SetBC(gmesh, 7, matIds[4]);

  return gmesh;
}

TPZGeoMesh* MeshingUtils::CreateGeoMesh3D(
  const TPZManVector<int, 3> &nelDiv, 
  const TPZManVector<REAL, 3> &minX, 
  const TPZManVector<REAL, 3> &maxX,
  const TPZManVector<int, 7> &matIds) {

  TPZGenGrid3D generator(minX, maxX, nelDiv, MMeshType::EHexahedral);

  generator.BuildVolumetricElements(matIds[0]);
  TPZGeoMesh *gmesh = generator.BuildBoundaryElements(matIds[1], matIds[2], matIds[3], matIds[4], matIds[5], matIds[6]);

  return gmesh;
}

TPZGeoMesh* MeshingUtils::ReadGeoMesh(std::string file, int EDomain, int EDirichlet, int EDirichlet2) {
  std::string currentPath = std::filesystem::current_path();
  std::string fatherPath = std::filesystem::path(currentPath).parent_path();
  std::string path(fatherPath + "/Inputs/" + file);
  TPZGeoMesh *gmesh = new TPZGeoMesh();
  {
    TPZGmshReader reader;
    TPZManVector<std::map<std::string, int>, 4> stringtoint(4);

    stringtoint[2]["Domain"] = EDomain;
    stringtoint[1]["Outer"] = EDirichlet;
    stringtoint[1]["Inner"] = EDirichlet2;

    reader.SetDimNamePhysical(stringtoint);
    reader.GeometricGmshMesh(path, gmesh);
  }

  // Remove gmsh boundary elements and create GeoElBC so normals are consistent
  int64_t nel = gmesh->NElements();
  for (int64_t el = 0; el < nel; el++) {
    TPZGeoEl *gel = gmesh->Element(el);
    if (!gel || gel->Dimension() != gmesh->Dimension() - 1) continue;
    TPZGeoElSide gelside(gel);
    TPZGeoElSide neigh = gelside.Neighbour();
    gel->RemoveConnectivities();
    int matid = gel->MaterialId();
    delete gel;
    TPZGeoElBC gbc(neigh, matid);
  }

  // Plot gmesh
  std::ofstream out("importedMesh.vtk");
  TPZVTKGeoMesh::PrintGMeshVTK(gmesh, out);

  return gmesh;
}