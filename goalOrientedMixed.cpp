#include "DarcyFlow/TPZDarcyFlow.h"
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "TPZAnalyticSolution.h"
#include "TPZGenGrid2D.h"
#include "TPZGenGrid3D.h"
#include "TPZGeoMeshTools.h"
#include "TPZGmshReader.h"
#include "TPZLinearAnalysis.h"
#include "TPZMultiphysicsCompMesh.h"
#include "TPZNullMaterial.h"
#include "TPZRefPatternDataBase.h"
#include "TPZSSpStructMatrix.h"
#include "TPZVTKGenerator.h"
#include "TPZVTKGeoMesh.h"
#include "pzintel.h"
#include "pzlog.h"
#include "pzmultiphysicselement.h"
#include "TPZHDivApproxCreator.h"
#include "pzcondensedcompel.h"
#include "pzstepsolver.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

// ================
// Global variables
// ================

int gthreads = 18;

// Exact solution
TLaplaceExample1 gexact;

// Permeability
REAL gperm = 1.0;

// Tolerance for error visualization
REAL gtol = 1e-3;
REAL ptol = 0.5;

// Material IDs for domain and boundaries
enum EnumMatIds { 
  EDomain = 1, 
  EFarfield = 2, 
  ECylinder = 3, 
  ETampa = 4,
  ECurveTampa = 5,
  EGoal = 6,
  ENone = -1 };

// ===================
// Function prototypes
// ===================

// Read mesh from .msh file
TPZGeoMesh *createGeoMesh(std::string file);

// Create geometric mesh using TPZGenGrid3D
TPZGeoMesh* createGeoMesh(
  const TPZManVector<int, 3> &nelDiv, 
  const TPZManVector<REAL, 3> &minX, 
  const TPZManVector<REAL, 3> &maxX);

  // Create geometric mesh using TPZGenGrid2D
// TPZGeoMesh* createGeoMesh(
//     const TPZManVector<int, 2> &nelDiv,
//     const TPZManVector<REAL, 2> &minX,
//     const TPZManVector<REAL, 2> &maxX);

// Computes the diameter of a geometric element
REAL ElementDiameter(TPZGeoEl *gel);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Create a computational mesh for the dual problem
TPZMultiphysicsCompMesh *createCompMeshMixedDual(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Error estimation function for H1 solution using mixed solution as reference
REAL GoalEstimation(TPZMultiphysicsCompMesh* cmeshMixed, TPZCompMesh* cmesh, int nthreads);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

// Initialize logger
#ifdef PZ_LOG
  TPZLogger::InitializePZLOG("logpz.txt");
#endif

  gperm = 1.0;

  // --- Solve darcy problem ---

  int mixed_order = 1; // Polynomial order

  // Set a problem with analytic solution
  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fDimension = 3;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  std::ofstream hrefFile("hrefTest.txt");

  // --- h-refinement loop ---

  int iteration = 0;
  TPZGeoMesh *gmesh = createGeoMesh({2, 2, 2}, {0., 0., 0.}, {1., 1., 1.});

  while (iteration < 6) {
    TPZMultiphysicsCompMesh *cmeshMixed =
        createCompMeshMixed(gmesh, mixed_order, true);
    TPZMultiphysicsCompMesh *cmeshDual =
        createCompMeshMixedDual(gmesh, mixed_order + 1, true);

    // Mixed solver
    TPZLinearAnalysis anMixed(cmeshMixed);
    TPZSSpStructMatrix<STATE> matMixed(cmeshMixed);
    matMixed.SetNumThreads(gthreads);
    anMixed.SetStructuralMatrix(matMixed);
    TPZStepSolver<STATE> stepMixed;
    stepMixed.SetDirect(ELDLt);
    anMixed.SetSolver(stepMixed);
    anMixed.Run();
    anMixed.SetThreadsForError(gthreads);

    // Dual solver
    TPZLinearAnalysis anDual(cmeshDual);
    TPZSSpStructMatrix<STATE> matDual(cmeshDual);
    matDual.SetNumThreads(gthreads);
    anDual.SetStructuralMatrix(matDual);
    TPZStepSolver<STATE> stepDual;
    stepDual.SetDirect(ELDLt);
    anDual.SetSolver(stepDual);
    anDual.Run();
    anDual.SetThreadsForError(gthreads);

    // --- Plotting ---

    {
      const std::string plotfile = "mixed_plot";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshMixed, fields, plotfile, vtkRes);
      vtk.Do();
    }

    {
      const std::string plotfile = "dual_plot";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshDual, fields, plotfile, vtkRes);
      vtk.Do();
    }

    // --- Goal-oriented error estimation ---

    REAL estimation = GoalEstimation(cmeshMixed, cmeshDual, gthreads);
    std::cout << "\nEstimated goal-oriented error: " << estimation << std::endl;

    // --- Clean up ---

    delete cmeshDual;
    delete cmeshMixed;
    iteration++;
  }
}

// =========
// Functions
// =========

TPZGeoMesh *createGeoMesh(std::string file) {

  std::string currentPath = std::filesystem::current_path();
  std::string fatherPath = std::filesystem::path(currentPath).parent_path();
  std::string path(fatherPath + "/" + file);
  TPZGeoMesh *gmesh = new TPZGeoMesh();
  {
    TPZGmshReader reader;
    TPZManVector<std::map<std::string, int>, 4> stringtoint(4);
    stringtoint[3]["volume_reservoir"] = EDomain;

    stringtoint[2]["surface_wellbore_cylinder"] = ECylinder;
    stringtoint[2]["surface_wellbore_heel"] = ETampa;
    stringtoint[2]["surface_wellbore_toe"] = ETampa;
    stringtoint[2]["surface_farfield"] = EFarfield;
    stringtoint[2]["surface_cap_rock"] = EFarfield;

    stringtoint[1]["curve_wellbore"] = ENone;
    stringtoint[1]["curve_heel"] = ECurveTampa;
    stringtoint[1]["curve_toe"] = ECurveTampa;

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

TPZGeoMesh* createGeoMesh(
  const TPZManVector<int, 3> &nelDiv, 
  const TPZManVector<REAL, 3> &minX, 
  const TPZManVector<REAL, 3> &maxX) {

  TPZGenGrid3D generator(minX, maxX, nelDiv, MMeshType::EHexahedral);

  generator.BuildVolumetricElements(EDomain);
  TPZGeoMesh *gmesh = generator.BuildBoundaryElements(EGoal, EFarfield, EFarfield, EFarfield, EFarfield, EFarfield);

  return gmesh;
}

TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order, bool isCondensed) {
  TPZHDivApproxCreator *hdivCreator = new TPZHDivApproxCreator(gmesh);
  hdivCreator->HdivFamily() = HDivFamily::EHDivStandard;
  hdivCreator->ProbType() = ProblemType::EDarcy;
  hdivCreator->SetDefaultOrder(order);
  hdivCreator->SetShouldCondense(isCondensed);
  hdivCreator->HybridType() = HybridizationType::ENone;

  TPZMixedDarcyFlow* matDarcy = new TPZMixedDarcyFlow(EDomain, gmesh->Dimension());
  matDarcy->SetConstantPermeability(gperm);
  matDarcy->SetForcingFunction(gexact.ForceFunc(), 4);
  matDarcy->SetExactSol(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(matDarcy);

  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 1.);
  
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EFarfield, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);

  bcond = matDarcy->CreateBC(matDarcy, EGoal, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);

  val2[0] = 0.;
  bcond = matDarcy->CreateBC(matDarcy, ECylinder, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);
  
  val2[0] = 0.;
  bcond = matDarcy->CreateBC(matDarcy, ETampa, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
  return cmesh;
}

TPZMultiphysicsCompMesh *createCompMeshMixedDual(TPZGeoMesh *gmesh, int order, bool isCondensed) {
  TPZHDivApproxCreator *hdivCreator = new TPZHDivApproxCreator(gmesh);
  hdivCreator->HdivFamily() = HDivFamily::EHDivStandard;
  hdivCreator->ProbType() = ProblemType::EDarcy;
  hdivCreator->SetDefaultOrder(order);
  hdivCreator->SetShouldCondense(isCondensed);
  hdivCreator->HybridType() = HybridizationType::ENone;

  TPZMixedDarcyFlow* matDarcy = new TPZMixedDarcyFlow(EDomain, gmesh->Dimension());
  matDarcy->SetConstantPermeability(gperm);
  hdivCreator->InsertMaterialObject(matDarcy);

  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 1.);
  
  val2[0] = 1.;
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EGoal, 0, val1, val2);
  hdivCreator->InsertMaterialObject(bcond);

  val2[0] = 0.;
  bcond = matDarcy->CreateBC(matDarcy, EFarfield, 0, val1, val2);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
  return cmesh;
}

REAL ElementDiameter(TPZGeoEl *gel) {
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

REAL GoalEstimation(TPZMultiphysicsCompMesh* cmesh, TPZCompMesh* cmeshDual, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread

  // Ensure references point to dual cmesh
  cmeshDual->Reference()->ResetReference();
  cmeshDual->LoadReferences();

  int64_t ncel = cmesh->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmesh->ElementVec();
  TPZVec<REAL> elementErrors(ncel, 0.0);

  // Parallelization setup
  std::vector<std::thread> threads(nthreads);
  TPZManVector<REAL> partialErrors(nthreads, 0.0);

  auto worker = [&](int tid, int64_t start, int64_t end) {
    REAL localTotalError = 0.0;
    for (int64_t icel = start; icel < end; ++icel) {
      TPZCompEl *celMixed = elementvec_m[icel];

      // Check if element is condensed
      TPZCondensedCompEl *condEl = dynamic_cast<TPZCondensedCompEl *>(celMixed);
      if (condEl) {
        // If compel is condensed, load solution on the unconsensed compel
        condEl->LoadSolution();
        celMixed = condEl->ReferenceCompEl();
      }

      int matid = celMixed->Material()->Id();
      if (matid != EDomain) continue;

      TPZGeoEl *gel = celMixed->Reference();
      TPZCompEl *celDual = gel->Reference();

      // Check if element is condensed
      TPZCondensedCompEl *condElDual = dynamic_cast<TPZCondensedCompEl *>(celDual);
      if (condElDual) {
        // If compel is condensed, load solution on the unconsensed compel
        condElDual->LoadSolution();
        celDual = condElDual->ReferenceCompEl();
      }

      if (!celMixed || !celDual) continue;
      if (gel->HasSubElement()) continue;
      if (celDual->Material()->Id() != matid) DebugStop();

      REAL hk = ElementDiameter(gel);
      REAL perm = gperm;
      REAL sqrtPerm = sqrt(perm);

      REAL goalError = 0.0;

      // Set integration rule
      const TPZIntPoints* intrule = nullptr;
      const TPZIntPoints &intruleMixed = celMixed->GetIntegrationRule();
      const TPZIntPoints &intruleDual = celDual->GetIntegrationRule();
      if (intruleMixed.NPoints() < intruleDual.NPoints()) {
        intrule = &intruleDual;
      } else {
        intrule = &intruleMixed;
      }

      for (int ip = 0; ip < intrule->NPoints(); ++ip) {
        TPZManVector<REAL,3> ptInElement(gel->Dimension());
        REAL weight, detjac;
        intrule->Point(ip, ptInElement, weight);
        TPZFNMatrix<9, REAL> jacobian, axes, jacinv;
        gel->Jacobian(ptInElement, jacobian, axes, detjac, jacinv);
        weight *= fabs(detjac);

        TPZManVector<REAL, 3> x(3, 0.0);
        gel->X(ptInElement, x); // Real coordinates for x

        TPZManVector<REAL,3> force(1,0.0);
        gexact.ForceFunc()(x, force); // Forcing function

        // Compute mixed solution ph and sigh
        TPZManVector<REAL,3> ph(gel->Dimension(),0.0);
        TPZManVector<REAL,3> sigh(gel->Dimension(),0.0);
        TPZManVector<REAL,1> divsigh(1,0.0);
        celMixed->Solution(ptInElement, 2, ph);
        celMixed->Solution(ptInElement, 1, sigh);
        celMixed->Solution(ptInElement, 5, divsigh);

        // Compute dual solution zh and psih
        TPZManVector<REAL,3> zh(gel->Dimension(),0.0);
        TPZManVector<REAL,3> psih(gel->Dimension(),0.0);
        TPZManVector<REAL,1> divpsih(1,0.0);
        celDual->Solution(ptInElement, 2, zh);
        celDual->Solution(ptInElement, 1, psih);
        celDual->Solution(ptInElement, 5, divpsih);

        // F contribution
        REAL FTerm = force[0]*zh[0];

        // Flux contribution
        REAL FluxTerm = 0.0;
        for (int d = 0; d < sigh.size(); ++d) {
          FluxTerm += (1./perm)*sigh[d]*psih[d];
        }

        // div contributions
        REAL divTermA = divsigh[0]*zh[0];
        REAL divTermB = divpsih[0]*ph[0];

        goalError += (FTerm - FluxTerm - divTermA - divTermB) * weight;
      }

      elementErrors[icel] = goalError;
      localTotalError += elementErrors[icel];
    }
    partialErrors[tid] = localTotalError;
  };

  int64_t chunk = ncel/nthreads;
  for (int t = 0; t < nthreads; ++t) {
    int64_t start = t * chunk;
    int64_t end = (t == nthreads - 1) ? ncel : (t + 1) * chunk;
    threads[t] = std::thread(worker, t, start, end);
  }
  for (auto& th : threads) th.join();

  REAL totalError = 0.0;
  for (auto val : partialErrors) totalError += val;

  // VTK output
  std::ofstream out_estimator("goalEstimation.vtk");
  TPZVTKGeoMesh::PrintCMeshVTK(cmesh, out_estimator, elementErrors, "EstimatedError");

  std::cout << "\nTotal estimated error: " << totalError << std::endl;

  // h-refinement based on estimator
  // REAL maxError = *std::max_element(elementErrors.begin(), elementErrors.end());
  REAL maxError = *std::max_element(elementErrors.begin(), elementErrors.end(),
    [](REAL a, REAL b) { return std::abs(a) < std::abs(b); });
  TPZVec<int64_t> needRefinement;
  for (int64_t i = 0; i < ncel; ++i) {
    if (abs(elementErrors[i]) > ptol*maxError) {
      TPZGeoEl *gel = elementvec_m[i]->Reference();
      TPZVec<TPZGeoEl *> pv;
      gel->Divide(pv);

      // Refine boundary
      int firstside = gel->FirstSide(2);
      int lastside = gel->FirstSide(3);
      for (int side = firstside; side < lastside; ++side) {
        TPZGeoElSide gelSide(gel, side);
        std::set<int> bcIds = {EFarfield, EGoal, ECylinder, ETampa, ECurveTampa};
        TPZGeoElSide neigh = gelSide.HasNeighbour(bcIds);
        if (neigh) {
          TPZGeoEl *neighGel = neigh.Element();
          if (neighGel->Dimension() != 2) DebugStop();
          TPZVec<TPZGeoEl *> pv2;
          neighGel->Divide(pv2);
        }
      }
    }
  }

  // Uniform refinement
  // TPZCheckGeom checkgeom(cmeshMixed->Reference());
  // checkgeom.UniformRefine(1);

  return totalError;
}

