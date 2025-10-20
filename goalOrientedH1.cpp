#include <iostream>
#include <thread>

// PZ includes
#include "DarcyFlow/TPZDarcyFlow.h"
#include "TPZAnalyticSolution.h"
#include "TPZLinearAnalysis.h"
#include "TPZNullMaterial.h"
#include "TPZSSpStructMatrix.h"
#include "TPZVTKGenerator.h"
#include "TPZVTKGeoMesh.h"
#include "pzlog.h"
#include "pzstepsolver.h"

// Local includes
#include "MeshingUtils.h"
#include "RefinementUtils.h"

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
REAL rtol = 0.5;
bool shouldPlot = true;

// Forcing function for the dual problem
auto ForcingFunctionDual = [](const TPZVec<REAL> &pt, TPZVec<STATE> &result) {
  result.Resize(1); // Ensure proper size
  REAL x = pt[0];
  REAL y = pt[1];

  result[0] = 0.;
  if (x >= 0.75 && x <= 0.875 && y >= 0.75 && y <= 0.875) {
    result[0] = 1. / 0.015625; // Area of the square
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
REAL GoalEstimation(TPZCompMesh* cmesh, TPZCompMesh* cmeshDual, TPZVec<int> &refinementIndicator, int nthreads);

// Compute the error functional
REAL ComputeFunctional(TPZCompMesh* cmesh, int gthreads);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

// Initialize logger
#ifdef PZ_LOG
  TPZLogger::InitializePZLOG("logpz.txt");
#endif

  // --- Solve darcy problem ---

  gexact.fExact = TLaplaceExample1::ESinSin;
  int dim = 2;
  gexact.fDimension = dim;
  gperm = 1.0;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  std::ofstream hrefFile("hrefTest.txt");

  // --- h-refinement loop ---

  int maxiter = 8;
  TPZVec<REAL> estimatedValues(maxiter);
  TPZVec<REAL> exactValues(maxiter);
  TPZVec<int> numberDofs(maxiter);
  TPZVec<int> refinementIndicator;

  int h1_order = 1; // Polynomial order

  TPZGeoMesh *gmesh = nullptr;
  if(dim == 2) {
    gmesh = MeshingUtils::CreateGeoMesh2D({4, 4}, {0., 0.}, {1., 1.});
  } else if (dim == 3) {
    gmesh = MeshingUtils::CreateGeoMesh3D({4, 4, 4}, {0., 0., 0.}, {1., 1., 1.});
  }

  for (int iteration = 0; iteration < maxiter; iteration++) {
    TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, h1_order);
    TPZCompMesh *cmeshDual = createCompMeshH1Dual(gmesh, h1_order + 1);

    // H1 solver
    TPZLinearAnalysis an(cmeshH1);
    TPZSSpStructMatrix<STATE> mat(cmeshH1);
    mat.SetNumThreads(gthreads);
    an.SetStructuralMatrix(mat);
    TPZStepSolver<STATE> stepH1;
    stepH1.SetDirect(ECholesky);
    an.SetSolver(stepH1);
    an.Run();
    an.SetThreadsForError(gthreads);
    numberDofs[iteration] = cmeshH1->NEquations();

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

    if (shouldPlot) {
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
    }

    // --- Goal-oriented error estimation ---

    estimatedValues[iteration] = GoalEstimation(cmeshH1, cmeshDual, refinementIndicator, gthreads);
    exactValues[iteration] = ComputeFunctional(cmeshH1, gthreads);

    RefinementUtils::MeshSmoothing(gmesh, refinementIndicator);
    RefinementUtils::AdaptiveRefinement(gmesh, refinementIndicator);
    // RefinementUtils::UniformRefinement(gmesh);

    // --- Clean up ---

    delete cmeshDual;
    delete cmeshH1;
  }

  // --- Output results ---
  std::cout << "\nNumber of DoFs, Estimated values, Real Values, Eff. Index\n";
  for (int i = 0; i < estimatedValues.NElements(); i++) {
  std::cout << numberDofs[i] << " & "
        << std::scientific << std::setprecision(4) << estimatedValues[i] << " & "
        << std::scientific << std::setprecision(4) << exactValues[i] << " & "
        << std::fixed << std::setprecision(3) << estimatedValues[i] / exactValues[i] << "\n";
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
  mat->SetForcingFunction(gexact.ForceFunc(), 6);
  mat->SetExactSol(gexact.ExactSolution(), 6);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  val2[0] = 0.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EBoundary, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 6);
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
  mat->SetForcingFunction(ForcingFunctionDual, 6);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  val2[0] = 0.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EBoundary, 0, val1, val2);
  cmesh->InsertMaterialObject(bcond);

  cmesh->AutoBuild();

  return cmesh;
}

REAL GoalEstimation(TPZCompMesh* cmesh, TPZCompMesh* cmeshDual, TPZVec<int> &refinementIndicator, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread

  // Ensure references point to dual cmesh
  cmeshDual->Reference()->ResetReference();
  cmeshDual->LoadReferences();

  // Resizes refinementIndicator and fill with zeros
  int64_t ngel = cmesh->Reference()->NElements();
  refinementIndicator.Resize(ngel);
  refinementIndicator.Fill(0);

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

        // A contribution
        REAL ATerm = 0.0;
        for (int d = 0; d < dph.size(); ++d) {
          ATerm += (1./perm)*dph[d]*dzh[d];
        }

        goalError += (FTerm - ATerm) * weight;
      }

      elementErrors[icel] = std::abs(goalError);
      localTotalError += goalError;
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
  if (shouldPlot) {
    std::ofstream out_estimator("goalEstimation.vtk");
    TPZVTKGeoMesh::PrintCMeshVTK(cmesh, out_estimator, elementErrors,
                                 "EstimatedError");
  }

  // Mark elements for refinement
  REAL maxError = *std::max_element(elementErrors.begin(), elementErrors.end());
  for (int64_t i = 0; i < ncel; ++i) {
    if (elementErrors[i] > rtol*maxError) {
      int64_t igeo = cmesh->Element(i)->Reference()->Index();
      refinementIndicator[igeo] = 1;
    }
  }

  return totalError;
}

REAL ComputeFunctional(TPZCompMesh* cmesh, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread

  int64_t ncel = cmesh->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmesh->ElementVec();

  // Parallelization setup
  std::vector<std::thread> threads(nthreads);
  TPZManVector<REAL> threadContributions(nthreads, 0.0);

  auto worker = [&](int tid, int64_t start, int64_t end) {
    REAL threadContribution = 0.0;
    for (int64_t icel = start; icel < end; ++icel) {
      TPZCompEl *cel = elementvec_m[icel];
      int matid = cel->Material()->Id();
      if (matid != EDomain) continue;
      TPZGeoEl *gel = cel->Reference();

      REAL elementContribution = 0.0;

      // Set integration rule
      const TPZIntPoints* intrule = gel->CreateSideIntegrationRule(gel->NSides() - 1, 10);

      for (int ip = 0; ip < intrule->NPoints(); ++ip) {
        TPZManVector<REAL,3> ptInElement(gel->Dimension());
        REAL weight, detjac;
        intrule->Point(ip, ptInElement, weight);
        TPZFNMatrix<9, REAL> jacobian, axes, jacinv;
        gel->Jacobian(ptInElement, jacobian, axes, detjac, jacinv);
        weight *= fabs(detjac);

        TPZManVector<REAL, 3> x(3, 0.0);
        gel->X(ptInElement, x); // Real coordinates for x

        TPZVec<STATE> force(1);
        ForcingFunctionDual(x, force);

        // Compute solution ph
        TPZManVector<REAL,1> ph(1,0.0);
        cel->Solution(ptInElement, 1, ph);

        // Compute exact solution pe
        TPZVec<STATE> pe(1);
        TPZFMatrix<REAL> dummyJac;
        gexact.ExactSolution()(x, pe, dummyJac);

        elementContribution += force[0]*(pe[0]-ph[0])*weight;
      }
      threadContribution += elementContribution;
    }
    threadContributions[tid] = threadContribution;
  };

  int64_t chunk = ncel/nthreads;
  for (int t = 0; t < nthreads; ++t) {
    int64_t start = t * chunk;
    int64_t end = (t == nthreads - 1) ? ncel : (t + 1) * chunk;
    threads[t] = std::thread(worker, t, start, end);
  }
  for (auto& th : threads) th.join();

  REAL totalValue = 0.0;
  for (auto val : threadContributions) totalValue += val;

  return totalValue;
}