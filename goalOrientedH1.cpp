#include "DarcyFlow/TPZDarcyFlow.h"
#include "MeshingUtils.h"
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "TPZAnalyticSolution.h"
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

int gthreads = 0;

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

// Forcing function for the dual problem
auto ForcingFunctionDual = [](const TPZVec<REAL> &pt, TPZVec<STATE> &result) {
  result.Resize(1); // Ensure proper size
  REAL x = pt[0];
  REAL y = pt[1];

  result[0] = 0.;
  if (x >= 0.75 && x <= 0.875 && y >= 0.75 && y <= 0.875) {
    result[0] = 1.;
  }
};

// ===================
// Function prototypes
// ===================

// Creates a computational mesh for mixed approximation
TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order = 1);

// Create a computational mesh for the dual problem
TPZCompMesh *createCompMeshH1Dual(TPZGeoMesh *gmesh, int order = 1);

// Error estimation function for H1 solution using mixed solution as reference
REAL GoalEstimation(TPZCompMesh* cmesh, TPZCompMesh* cmeshDual, int nthreads);

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

  int h1_order = 1; // Polynomial order

  // Set a problem with analytic solution
  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fDimension = 2;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  std::ofstream hrefFile("hrefTest.txt");

  // --- h-refinement loop ---

  int iteration = 0;
  int maxiter = 8;
  TPZVec<REAL> estimatedValues(maxiter);
  TPZGeoMesh *gmesh = MeshingUtils::CreateGeoMesh2D({2, 2}, {0., 0.}, {1., 1.});

  while (iteration < maxiter) {
    TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, h1_order);
    TPZCompMesh *cmeshDual = createCompMeshH1Dual(gmesh, h1_order + 1);

    // Mixed solver
    TPZLinearAnalysis an(cmeshH1);
    TPZSSpStructMatrix<STATE> mat(cmeshH1);
    mat.SetNumThreads(gthreads);
    an.SetStructuralMatrix(mat);
    TPZStepSolver<STATE> stepMixed;
    stepMixed.SetDirect(ECholesky);
    an.SetSolver(stepMixed);
    an.Run();
    an.SetThreadsForError(gthreads);

    // Dual solver
    TPZLinearAnalysis anDual(cmeshDual);
    TPZSSpStructMatrix<STATE> matDual(cmeshDual);
    matDual.SetNumThreads(gthreads);
    anDual.SetStructuralMatrix(matDual);
    TPZStepSolver<STATE> stepDual;
    stepDual.SetDirect(ECholesky);
    anDual.SetSolver(stepDual);
    anDual.Run();
    anDual.SetThreadsForError(gthreads);

    // --- Plotting ---

    {
      const std::string plotfile = "h1_plot";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshH1, fields, plotfile, vtkRes);
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

    REAL estimation = GoalEstimation(cmeshH1, cmeshDual, gthreads);
    estimatedValues[iteration] = estimation;

    // --- Clean up ---

    delete cmeshDual;
    delete cmeshH1;
    iteration++;
  }
  // --- Output results ---
  std::cout << "\nEstimated values:\n";
  for (int i = 0; i < estimatedValues.NElements(); i++) {
    std::cout << estimatedValues[i] << "\n";
  }
}

// =========
// Functions
// =========

TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order) {
  TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
  cmesh->SetDimModel(gmesh->Dimension());
  cmesh->SetDefaultOrder(order);            // Polynomial order
  cmesh->SetAllCreateFunctionsContinuous(); // H1 Elements

  // Add materials (weak formulation)
  TPZDarcyFlow *mat = new TPZDarcyFlow(EDomain, gmesh->Dimension());
  mat->SetConstantPermeability(gperm);
  mat->SetForcingFunction(gexact.ForceFunc(), 8);
  mat->SetExactSol(gexact.ExactSolution(), 8);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  val2[0] = 1.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EFarfield, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 8);
  cmesh->InsertMaterialObject(bcond);

  cmesh->AutoBuild();

  return cmesh;
}

TPZCompMesh *createCompMeshH1Dual(TPZGeoMesh *gmesh, int order) {
  TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
  cmesh->SetDimModel(gmesh->Dimension());
  cmesh->SetDefaultOrder(order);            // Polynomial order
  cmesh->SetAllCreateFunctionsContinuous(); // H1 Elements

  // Add materials (weak formulation)
  TPZDarcyFlow *mat = new TPZDarcyFlow(EDomain, gmesh->Dimension());
  mat->SetConstantPermeability(gperm);
  mat->SetForcingFunction(ForcingFunctionDual, 8);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  val2[0] = 0.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EFarfield, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  cmesh->AutoBuild();

  return cmesh;
}

REAL GoalEstimation(TPZCompMesh* cmesh, TPZCompMesh* cmeshDual, int nthreads) {

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
      TPZCompEl *cel = elementvec_m[icel];
      int matid = cel->Material()->Id();
      if (matid != EDomain) continue;
      TPZGeoEl *gel = cel->Reference();
      TPZCompEl *celDual = gel->Reference();

      if (!cel || !celDual) continue;
      if (gel->HasSubElement()) continue;
      if (celDual->Material()->Id() != matid) DebugStop();

      REAL hk = MeshingUtils::ElementDiameter(gel);
      REAL perm = gperm;
      REAL sqrtPerm = sqrt(perm);

      REAL goalError = 0.0;

      // Set integration rule
      const TPZIntPoints* intrule = nullptr;
      const TPZIntPoints &intruleH1 = cel->GetIntegrationRule();
      const TPZIntPoints &intruleDual = celDual->GetIntegrationRule();
      if (intruleH1.NPoints() < intruleDual.NPoints()) {
        intrule = &intruleDual;
      } else {
        intrule = &intruleH1;
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

        // Compute mixed solution ph and dph
        TPZManVector<REAL,1> ph(1,0.0);
        TPZManVector<REAL,3> dph(gel->Dimension(),0.0);
        cel->Solution(ptInElement, 1, ph);
        cel->Solution(ptInElement, 2, dph);

        // Compute dual solution zh and psih
        TPZManVector<REAL,1> zh(1,0.0);
        TPZManVector<REAL,3> dzh(gel->Dimension(),0.0);
        celDual->Solution(ptInElement, 1, zh);
        celDual->Solution(ptInElement, 2, dzh);

        // F contribution
        REAL FTerm = force[0]*zh[0];

        // Flux contribution
        REAL FluxTerm = 0.0;
        for (int d = 0; d < dph.size(); ++d) {
          FluxTerm += (1./perm)*dph[d]*dzh[d];
        }

        goalError += (FTerm - FluxTerm) * weight;
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
        std::set<int> bcIds = {EFarfield};
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

