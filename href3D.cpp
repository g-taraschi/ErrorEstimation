#include "DarcyFlow/TPZDarcyFlow.h"
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "TPZAnalyticSolution.h"
#include "TPZGenGrid2D.h"
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
#include "pzstepsolver.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

// ================
// Global variables
// ================

int gthreads = 16;

// Exact solution
TLaplaceExample1 gexact;

// Permeability
REAL gperm = 1.0;

// Tolerance for error visualization
REAL gtol = 1.e-3;

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

// Computes the diameter of a geometric element
REAL ElementDiameter(TPZGeoEl *gel);

// Creates a computational mesh for H1 approximation
TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order = 1);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1);

// Error estimation function for H1 solution using mixed solution as reference
REAL ErrorEstimationH1(TPZCompMesh *cmesh, TPZMultiphysicsCompMesh *cmeshMixed, int iteration);

// Error estimation function for mixed solution using H1 solution as reference
REAL ErrorEstimationMixed(TPZCompMesh *cmesh,
                          TPZMultiphysicsCompMesh *cmeshMixed, int iteration);

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

  int order = 1; // Polynomial order

  // Set a problem with analytic solution
  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  // Initial mesh
  TPZGeoMesh *gmesh = createGeoMesh("mesh3D.msh");

  // --- h-refinement loop ---

  for (int iteration = 0; iteration < 3; iteration++) {

    TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, order);
    TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, order);

    // Mixed solver
    TPZLinearAnalysis anMixed(cmeshMixed);
    TPZSSpStructMatrix<STATE> matMixed(cmeshMixed);
    matMixed.SetNumThreads(gthreads);
    anMixed.SetStructuralMatrix(matMixed);
    TPZStepSolver<STATE> stepMixed;
    stepMixed.SetDirect(ELDLt);
    anMixed.SetSolver(stepMixed);
    anMixed.Run();

    // H1 solver
    TPZLinearAnalysis anH1(cmeshH1);
    TPZSSpStructMatrix<STATE> matH1(cmeshH1);
    matH1.SetNumThreads(gthreads);
    anH1.SetStructuralMatrix(matH1);
    TPZStepSolver<STATE> stepH1;
    stepH1.SetDirect(ECholesky);
    anH1.SetSolver(stepH1);
    anH1.Run();

    // ---- Plotting ---

    {
      // std::string filename = "geomesh" + std::to_string(iteration) + ".vtk";
      std::string filename = "geomesh.vtk";
      std::ofstream out(filename);
      TPZVTKGeoMesh::PrintGMeshVTK(gmesh, out);
    }

    {
      //std::string filename = "h1_plot_iter" + std::to_string(iteration) + ".vtk";
      std::string filename = "h1_plot_iter";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshH1, fields, filename, vtkRes);
      vtk.Do();
    }

    {
      // std::string filename = "mixed_plot_iter" + std::to_string(iteration) + ".vtk";
      std::string filename = "mixed_plot_iter";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshMixed, fields, filename, vtkRes);
      vtk.Do();
    }

    // --- Error Estimation ---

    std::cout << "Estimating errors...\n" << std::endl;

    REAL EstimatedErrorH1 = ErrorEstimationH1(cmeshH1, cmeshMixed, iteration);
    // REAL EstimatedErrorMixed = ErrorEstimationMixed(cmeshH1, cmeshMixed, iteration);

    std::cout << "Iteration " << iteration << ":\n"
              << "    Estimated Error for H1 = " << EstimatedErrorH1 << "\n"
              << "\n"
              << std::endl;

    std::ofstream file("hrefTest.txt");
    file << EstimatedErrorH1 << std::endl;

    // --- Clean up ---

    // delete cmeshH1, cmeshMixed;
    delete cmeshH1;
    delete cmeshMixed;
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
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  val2[0] = 1.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EFarfield, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 2.0;
  bcond = mat->CreateBC(mat, ECylinder, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 0.0;
  bcond = mat->CreateBC(mat, ETampa, 1, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 2.0;
  bcond = mat->CreateBC(mat, ECurveTampa, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  // Set up the computational mesh
  cmesh->AutoBuild();

  return cmesh;
}

TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order) {

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
  cmesh->InsertMaterialObject(matDarcy);

  // Create, set and add boundary conditions
  val2[0] = 1.0;
  bcond = matDarcy->CreateBC(matDarcy, EFarfield, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 2.0;
  bcond = matDarcy->CreateBC(matDarcy, ECylinder, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  val2[0] = 0.0;
  bcond = matDarcy->CreateBC(matDarcy, ETampa, 1, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  // Incorporate the atomic meshes into the multiphysics mesh
  TPZManVector<TPZCompMesh *, 2> cmeshes(2);
  cmeshes[0] = cmeshFlux;
  cmeshes[1] = cmeshPressure;

  TPZManVector<int> active(cmeshes.size(), 1);
  cmesh->BuildMultiphysicsSpace(active, cmeshes);

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

REAL ErrorEstimationH1(TPZCompMesh *cmesh, TPZMultiphysicsCompMesh *cmeshMixed, int iteration) {
  // Ensure references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();
  int dim = cmesh->Dimension();

  int64_t ncel = cmeshMixed->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();
  TPZVec<REAL> elementErrors(ncel, 0.0);

  // Parallelization setup
  std::vector<std::thread> threads(gthreads);
  TPZManVector<REAL> partialErrors(gthreads, 0.0);

  auto worker = [&](int tid, int64_t start, int64_t end) {
    REAL localTotalError = 0.0;
    for (int64_t icel = start; icel < end; ++icel) {
      TPZCompEl *celMixed = elementvec_m[icel];
      TPZGeoEl *gel = celMixed->Reference();
      TPZCompEl *celH1 = gel->Reference();

      if (!gel || gel->Dimension() != dim || gel->HasSubElement())
        continue;
      if (!celMixed || !celH1)
        continue;

      REAL hk = ElementDiameter(gel);
      REAL sqrtPerm = sqrt(gperm);

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

        TPZManVector<REAL,3> force(1,0.0);

        // Compute H1 term K^(1/2) grad(u)
        TPZManVector<REAL,3> termH1(gel->Dimension(),0.0);
        celH1->Solution(ptInElement, 2, termH1);
        for (int d = 0; d < dim; ++d) {
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
        for (int d = 0; d < dim; ++d) {
          termHdiv[d] = (-1./sqrtPerm) * termHdiv[d];
        }

        // Flux contribution
        REAL diffFlux = 0.0;
        for (int d = 0; d < dim; ++d) {
          REAL diff = termH1[d] - termHdiv[d];
          diffFlux += diff * diff;
        }

        // Balance contribution
        REAL diffBalance = (divFluxMixed[0] - force[0]) * (divFluxMixed[0] - force[0]);

        fluxError += diffFlux * weight;
        balanceError += diffBalance * weight;
      }

      REAL contribution = sqrt(fluxError) + (hk/(M_PI*sqrt(gperm)))*sqrt(balanceError);
      elementErrors[icel] = (contribution * contribution);
      localTotalError += elementErrors[icel];
    }
    partialErrors[tid] = localTotalError;
  };

  int64_t chunk = ncel / gthreads;
  for (int t = 0; t < gthreads; ++t) {
    int64_t start = t * chunk;
    int64_t end = (t == gthreads - 1) ? ncel : (t + 1) * chunk;
    threads[t] = std::thread(worker, t, start, end);
  }
  for (auto& th : threads) th.join();

  REAL totalError = 0.0;
  for (auto val : partialErrors) totalError += val;
  
  totalError = sqrt(totalError);

  // VTK output
  // std::string filename = "EstimatedErrorMixed_iter" + std::to_string(iteration) + ".vtk";
  std::string filename = "EstimatedErrorH1_iter.vtk";
  std::ofstream out_estimator(filename);
  TPZVTKGeoMesh::PrintCMeshVTK(cmeshMixed, out_estimator, elementErrors, "EstimatedErrorH1");

  // h-refinement
  int count = 0;
  for (int64_t icel = 0; icel < elementErrors.size(); ++icel) {
    if (elementErrors[icel] > gtol) {
      count++;
      TPZGeoEl *gel = cmeshMixed->Element(icel)->Reference();
      if (!gel) DebugStop();
      if (gel->HasSubElement()) continue; // We have to do this because of duplicated entries
      TPZVec<TPZGeoEl *> pv;
      TPZCheckGeom checkgeom(cmeshMixed->Reference());
      gel->Divide(pv);
      int firstside = gel->FirstSide(2);
      int lastside = gel->FirstSide(3);
      for (int side = firstside; side < lastside; side++) {
        TPZGeoElSide gelside(gel, side);
        TPZGeoElSide neigh = gelside.Neighbour();
        if (neigh.Element()->Dimension() == 2) {
          TPZVec<TPZGeoEl *> pv2;
          neigh.Element()->Divide(pv2);
        }
      }
    }
  }

  std::cout << "    Number of elements refined = " << count << "\n"
            << std::endl;
  return totalError;
}

REAL ErrorEstimationMixed(TPZCompMesh *cmesh, TPZMultiphysicsCompMesh *cmeshMixed, int iteration) {
  // Ensure references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();
  int dim = cmesh->Dimension();

  int64_t ncel = cmeshMixed->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();
  TPZVec<REAL> elementErrors(ncel, 0.0);

  // Parallelization setup
  std::vector<std::thread> threads(gthreads);
  TPZManVector<REAL> partialErrors(gthreads, 0.0);

  auto worker = [&](int tid, int64_t start, int64_t end) {
    REAL localTotalError = 0.0;
    for (int64_t icel = start; icel < end; ++icel) {
      TPZCompEl *celMixed = elementvec_m[icel];
      TPZGeoEl *gel = celMixed->Reference();
      TPZCompEl *celH1 = gel->Reference();

      if (!gel || gel->Dimension() != dim || gel->HasSubElement())
        continue;
      if (!celMixed || !celH1)
        continue;

      REAL hk = ElementDiameter(gel);
      REAL sqrtPerm = sqrt(gperm);

      REAL fluxError = 0.0;
      REAL balanceError = 0.0;
      REAL conformityError = 0.0;

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

        TPZManVector<REAL,3> force(1,0.0);

        // Compute H1 term K^(1/2) grad(u)
        TPZManVector<REAL,3> termH1(gel->Dimension(),0.0);
        celH1->Solution(ptInElement, 2, termH1);
        for (int d = 0; d < dim; ++d) {
          termH1[d] = sqrtPerm * termH1[d];
        }

        // Compute L2 term K^(1/2) grad(u)
        TPZManVector<REAL,3> termL2(3,0.0);
        celMixed->Solution(ptInElement, 9, termL2);
        for (int d = 0; d < dim; ++d) {
          termL2[d] = sqrtPerm * termL2[d];
        }

        // Compute Hdiv term -K^(-1/2) sig and div(sig)
        TPZManVector<REAL,3> termHdiv(3,0.0);
        TPZManVector<REAL,1> divFluxMixed(1,0.0);
        TPZMultiphysicsElement *celMulti = dynamic_cast<TPZMultiphysicsElement*>(celMixed);
        if (celMulti) {
          celMulti->Solution(ptInElement, 1, termHdiv);
          celMulti->Solution(ptInElement, 5, divFluxMixed);
        }
        for (int d = 0; d < dim; ++d) {
          termHdiv[d] = (-1./sqrtPerm) * termHdiv[d];
        }

        // Flux contribution
        REAL diffFlux = 0.0;
        for (int d = 0; d < dim; ++d) {
          REAL diff = termL2[d] - termHdiv[d];
          diffFlux += diff * diff;
        }

        // Balance contribution
        REAL diffBalance = (divFluxMixed[0] - force[0]) * (divFluxMixed[0] - force[0]);

        // Conformity contribution
        REAL diffConformity = 0.0;
        for (int d = 0; d < dim; ++d) {
          REAL diff = termH1[d] - termL2[d];
          diffConformity += diff * diff;
        }

        fluxError += diffFlux * weight;
        balanceError += diffBalance * weight;
        conformityError += diffConformity * weight;
      }

      REAL contribution = sqrt(fluxError) + (hk/(M_PI*sqrtPerm))*sqrt(balanceError);
      elementErrors[icel] = (contribution * contribution) + conformityError;
      localTotalError += elementErrors[icel];
    }
    partialErrors[tid] = localTotalError;
  };

  int64_t chunk = ncel / gthreads;
  for (int t = 0; t < gthreads; ++t) {
    int64_t start = t * chunk;
    int64_t end = (t == gthreads - 1) ? ncel : (t + 1) * chunk;
    threads[t] = std::thread(worker, t, start, end);
  }
  for (auto& th : threads) th.join();

  REAL totalError = 0.0;
  for (auto val : partialErrors) totalError += val;
  
  totalError = sqrt(totalError);

  // VTK output
  std::string filename = "EstimatedErrorH1_iter" + std::to_string(iteration) + ".vtk";
  std::ofstream out_estimator(filename);
  TPZVTKGeoMesh::PrintCMeshVTK(cmeshMixed, out_estimator, elementErrors, "EstimatedErrorH1");

  return totalError;
}