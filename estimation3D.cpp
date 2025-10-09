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
  ENone = -1 };

// ===================
// Function prototypes
// ===================

// Creates a geometric mesh using TPZGenGrid2D
TPZGeoMesh *createGeoMesh(std::string file);

TPZGeoMesh* createGeoMesh(
  const TPZManVector<int, 3> &nelDiv, 
  const TPZManVector<REAL, 3> &minX, 
  const TPZManVector<REAL, 3> &maxX);

// Computes the diameter of a geometric element
REAL ElementDiameter(TPZGeoEl *gel);

// Creates a computational mesh for H1 approximation
TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order = 1);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixedOld(TPZGeoMesh *gmesh, int order = 1);

// Error estimation function for H1 solution using mixed solution as reference
REAL ErrorEstimation(TPZMultiphysicsCompMesh* cmeshMixed, TPZCompMesh* cmesh, int nthreads);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

// Initialize logger
#ifdef PZ_LOG
  TPZLogger::InitializePZLOG("logpz.txt");
#endif

  gperm = 3.0;

  // --- Solve darcy problem ---

  int mixed_order = 1; // Polynomial order
  int h1_order = 1;

  // Set a problem with analytic solution
  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fDimension = 3;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  std::ofstream hrefFile("hrefTest.txt");

  // --- h-refinement loop ---

  while (h1_order < 4) {
    hrefFile << "\n\nOrder H1: " << h1_order << ", Order Mixed: " << mixed_order
             << "\n";
    // Initial mesh
    // TPZGeoMesh *gmesh = createGeoMesh("mesh3D.msh");
    TPZGeoMesh *gmesh = createGeoMesh({2, 2, 2}, {0., 0., 0.}, {1., 1., 1.});
    int iteration = 0;
    REAL estimatedError = 1.;
    while (iteration < 6) {
      TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, mixed_order, true);
      TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, h1_order);

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

      // H1 solver
      TPZLinearAnalysis anH1(cmeshH1);
      TPZSSpStructMatrix<STATE> matH1(cmeshH1);
      matH1.SetNumThreads(gthreads);
      anH1.SetStructuralMatrix(matH1);
      TPZStepSolver<STATE> stepH1;
      stepH1.SetDirect(ECholesky);
      anH1.SetSolver(stepH1);
      anH1.Run();
      anH1.SetThreadsForError(gthreads);

      // ---- Plotting ---

      // {
      //   const std::string plotfile = "h1_plot";
      //   constexpr int vtkRes{0};
      //   TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      //   auto vtk = TPZVTKGenerator(cmeshH1, fields, plotfile, vtkRes);
      //   vtk.Do();
      // }

      // {
      //   const std::string plotfile = "mixed_plot";
      //   constexpr int vtkRes{0};
      //   TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      //   auto vtk = TPZVTKGenerator(cmeshMixed, fields, plotfile, vtkRes);
      //   vtk.Do();
      // }

      // --- Error Estimation ---

      // Exact error for H1 solution
      TPZVec<REAL> errorsH1(3, 0.);
      anH1.PostProcessError(errorsH1, false, std::cout);

      // Exact error for mixed solution
      TPZVec<REAL> errorsMixed(5, 0.);
      anMixed.PostProcessError(errorsMixed, false, std::cout);
      errorsMixed[1] = (1. / sqrt(gperm)) * errorsMixed[1];

      estimatedError = ErrorEstimation(cmeshMixed, cmeshH1, gthreads);
      REAL PgSyDiff = estimatedError*estimatedError - errorsH1[0]*errorsH1[0] - errorsMixed[1]*errorsMixed[1];

      // Print results
      std::cout << "\nIteration " << iteration << ":\n"
                << "    Estimated Error for H1 = " << estimatedError
                << ", Actual error for H1 = " << errorsH1[0]
                << ", Effective index H1 = " << estimatedError / errorsH1[0]
                << "\n"
                << "    Estimated Error for Mixed = " << estimatedError
                << ", Actual error for Mixed = " << errorsMixed[1]
                << ", Effective index Mixed = "
                << estimatedError / errorsMixed[1] << std::endl;

      hrefFile << std::scientific << std::setprecision(3) << estimatedError
               << " & " << errorsH1[0] << " & " << estimatedError / errorsH1[0]
               << " & " << estimatedError << " & " << errorsMixed[1] << " & "
               << estimatedError / errorsMixed[1] << " & " << PgSyDiff << std::endl;

      iteration++;

      // --- Clean up ---

      delete cmeshH1;
      delete cmeshMixed;
    }
    h1_order++;
    delete gmesh;
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
  TPZGeoMesh *gmesh = generator.BuildBoundaryElements(EFarfield, EFarfield, EFarfield, EFarfield, EFarfield, EFarfield);

  return gmesh;
}

TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order) {
  TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
  cmesh->SetDimModel(gmesh->Dimension());
  cmesh->SetDefaultOrder(order);            // Polynomial order
  cmesh->SetAllCreateFunctionsContinuous(); // H1 Elements

  // Add materials (weak formulation)
  TPZDarcyFlow *mat = new TPZDarcyFlow(EDomain, gmesh->Dimension());
  mat->SetConstantPermeability(gperm);
  mat->SetForcingFunction(gexact.ForceFunc(), 4);
  mat->SetExactSol(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  val2[0] = 1.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EFarfield, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 2.0;
  bcond = mat->CreateBC(mat, ECylinder, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 0.0;
  bcond = mat->CreateBC(mat, ETampa, 1, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  // val2[0] = 0.0;
  // bcond = mat->CreateBC(mat, ECurveTampa, 1, val1, val2);
  // // bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  // cmesh->InsertMaterialObject(bcond);

  // Set up the computational mesh
  cmesh->AutoBuild();

  return cmesh;
}

TPZMultiphysicsCompMesh *createCompMeshMixedOld(TPZGeoMesh *gmesh, int order) {

  // --- Flux atomic cmesh ----

  TPZCompMesh *cmeshFlux = new TPZCompMesh(gmesh);
  cmeshFlux->SetDimModel(gmesh->Dimension());
  cmeshFlux->SetDefaultOrder(order);
  cmeshFlux->SetAllCreateFunctionsHDiv();

  // Add materials (weak formulation)
  TPZNullMaterial<STATE> *mat =
      new TPZNullMaterial(EDomain, gmesh->Dimension());
  cmeshFlux->InsertMaterialObject(mat);

  // Create boundary conditions
  TPZManVector<REAL, 1> val2(1, 0.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EFarfield, 0, val1, val2);
  cmeshFlux->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ECylinder, 0, val1, val2);
  cmeshFlux->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ETampa, 0, val1, val2);
  cmeshFlux->InsertMaterialObject(bcond);

  cmeshFlux->AutoBuild();

  // --- Pressure atomic cmesh ---

  TPZCompMesh *cmeshPressure = new TPZCompMesh(gmesh);
  cmeshPressure->SetDimModel(gmesh->Dimension());
  cmeshPressure->SetDefaultOrder(order);
  if (order < 1) {
    cmeshPressure->SetAllCreateFunctionsDiscontinuous();
  } else {
    cmeshPressure->SetAllCreateFunctionsContinuous();
    cmeshPressure->ApproxSpace().CreateDisconnectedElements(true);
  }

  // Add materials (weak formulation)
  cmeshPressure->InsertMaterialObject(mat);

  // Set up the computational mesh
  cmeshPressure->AutoBuild();

  int ncon = cmeshPressure->NConnects();
  const int lagLevel = 1; // Lagrange multiplier level
  for (int i = 0; i < ncon; i++) {
    TPZConnect &newnod = cmeshPressure->ConnectVec()[i];
    newnod.SetLagrangeMultiplier(lagLevel);
  }

  // --- Multiphysics mesh ---

  TPZMultiphysicsCompMesh *cmesh = new TPZMultiphysicsCompMesh(gmesh);
  cmesh->SetDimModel(gmesh->Dimension());
  cmesh->SetDefaultOrder(1);
  cmesh->ApproxSpace().Style() = TPZCreateApproximationSpace::EMultiphysics;

  // Add materials (weak formulation)
  TPZMixedDarcyFlow *matDarcy =
      new TPZMixedDarcyFlow(EDomain, gmesh->Dimension());
  matDarcy->SetConstantPermeability(gperm);
  matDarcy->SetForcingFunction(gexact.ForceFunc(), 4);
  matDarcy->SetExactSol(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(matDarcy);

  // Create, set and add boundary conditions
  val2[0] = 1.0;
  bcond = matDarcy->CreateBC(matDarcy, EFarfield, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 2.0;
  bcond = matDarcy->CreateBC(matDarcy, ECylinder, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 0.0;
  bcond = matDarcy->CreateBC(matDarcy, ETampa, 1, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  // Incorporate the atomic meshes into the multiphysics mesh
  TPZManVector<TPZCompMesh *, 2> cmeshes(2);
  cmeshes[0] = cmeshFlux;
  cmeshes[1] = cmeshPressure;

  TPZManVector<int> active(cmeshes.size(), 1);
  cmesh->BuildMultiphysicsSpace(active, cmeshes);

  return cmesh;
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
  
  val2[0] = 1.;
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EFarfield, 0, val1, val2);
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

REAL ErrorEstimation(TPZMultiphysicsCompMesh* cmeshMixed, TPZCompMesh* cmesh, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread

  // Ensure references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();

  int64_t ncel = cmeshMixed->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();
  TPZVec<REAL> elementErrors(ncel, 0.0);
  int aux = elementvec_m.NElements();

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
      TPZCompEl *celH1 = gel->Reference();
      if (!celMixed || !celH1) continue;
      if (gel->HasSubElement()) continue;
      if (celH1->Material()->Id() != matid) DebugStop();

      REAL hk = ElementDiameter(gel);
      REAL perm = gperm;
      REAL sqrtPerm = sqrt(perm);

      REAL fluxError = 0.0;
      REAL balanceError = 0.0;

      // Set integration rule
      const TPZIntPoints* intrule = nullptr;
      const TPZIntPoints &intruleMixed = celMixed->GetIntegrationRule();
      const TPZIntPoints &intruleH1 = celH1->GetIntegrationRule();
      if (intruleMixed.NPoints() < intruleH1.NPoints()) {
        intrule = &intruleH1;
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
        gexact.ForceFunc()(x, force); // Exact divergence (forcing function)


        // Compute H1 term K^(1/2) grad(u)
        TPZManVector<REAL,3> termH1(gel->Dimension(),0.0);
        celH1->Solution(ptInElement, 2, termH1);
        for (int d = 0; d < termH1.size(); ++d) {
          termH1[d] = sqrtPerm * termH1[d];
        }

        // Compute Hdiv term -K^(-1/2) sig and div(sig)
        TPZManVector<REAL,3> termHdiv(3,0.0);
        TPZManVector<REAL,1> divFluxMixed(1,0.0);
        TPZMultiphysicsElement *celMulti = dynamic_cast<TPZMultiphysicsElement*>(celMixed);
        if (celMulti) {
          celMulti->Solution(ptInElement, 1, termHdiv);
          celMulti->Solution(ptInElement, 5, divFluxMixed);
        }
        for (int d = 0; d < termHdiv.size(); ++d) {
          termHdiv[d] = (-1./sqrtPerm) * termHdiv[d];
        }

        // Flux contribution
        REAL diffFlux = 0.0;
        for (int d = 0; d < termH1.size(); ++d) {
          REAL diff = termH1[d] - termHdiv[d];
          diffFlux += diff * diff;
        }

        // Balance contribution
        REAL diffBalance = (divFluxMixed[0] - force[0]) * (divFluxMixed[0] - force[0]);

        fluxError += diffFlux * weight;
        balanceError += diffBalance * weight;
      }

      REAL contribution = sqrt(fluxError) + (hk/(M_PI*sqrtPerm))*sqrt(balanceError);
      elementErrors[icel] = (contribution * contribution);

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
  
  totalError = sqrt(totalError);

  // VTK output
  std::ofstream out_estimator("EstimatedError.vtk");
  TPZVTKGeoMesh::PrintCMeshVTK(cmeshMixed, out_estimator, elementErrors, "EstimatedError");

  std::cout << "\nTotal estimated error: " << totalError << std::endl;

  // h-refinement based on estimator
  REAL maxError = *std::max_element(elementErrors.begin(), elementErrors.end());
  TPZVec<int64_t> needRefinement;
  for (int64_t i = 0; i < ncel; ++i) {
    if (elementErrors[i] > ptol*maxError) {
      TPZGeoEl *gel = elementvec_m[i]->Reference();
      TPZVec<TPZGeoEl *> pv;
      gel->Divide(pv);

      // Refine boundary
      int firstside = gel->FirstSide(2);
      int lastside = gel->FirstSide(3);
      for (int side = firstside; side < lastside; ++side) {
        TPZGeoElSide gelSide(gel, side);
        std::set<int> bcIds = {EFarfield, ECylinder, ETampa, ECurveTampa};
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

