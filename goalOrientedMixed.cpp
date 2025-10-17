#include <iostream>
#include <thread>

// PZ includes
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "TPZMultiphysicsCompMesh.h"
#include "TPZHDivApproxCreator.h"
#include "pzcondensedcompel.h"
#include "TPZAnalyticSolution.h"
#include "TPZLinearAnalysis.h"
#include "TPZNullMaterial.h"
#include "TPZSSpStructMatrix.h"
#include "TPZVTKGenerator.h"
#include "TPZVTKGeoMesh.h"
#include "pzlog.h"
#include "pzmultiphysicscompel.h"
#include "pzstepsolver.h"

// Local includes
#include "MeshingUtils.h"
#include "RefinementUtils.h"

// ================
// Global variables
// ================

int gthreads = 17;
bool shouldPlot = true;

// Exact solution
TLaplaceExample1 gexact;
int testCase = 1;

// Permeability
REAL gperm = 1.0;

// Tolerance for error visualization
REAL gtol = 1e-3;
REAL rtol = 0.8;

// Forcing function for the dual problem
auto DirichletFunctionDual = [](const TPZVec<REAL> &pt, TPZVec<STATE> &result, TPZFMatrix<STATE> &deriv) {
  result.Resize(1);   // Ensure proper size
  deriv.Resize(3, 1); // 2D gradient but needs 3 components

  REAL x = pt[0];
  REAL y = pt[1];

  result[0] = 0.;
  if (x > -1e-6 && x < 1e-6 && y < 0.6 && y > 0.4) {
    result[0] = 1.0/0.2; // Length of curve
  }

  // Dummy argument, not used in this application
  deriv(0, 0) = 0.0; // du/dx
  deriv(1, 0) = 0.0; // du/dy
  deriv(2, 0) = 0.0; // empty
};

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
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Create a computational mesh for the dual problem
TPZMultiphysicsCompMesh *createCompMeshMixedDual(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Create a computational mesh for the dual problem
TPZMultiphysicsCompMesh *createCompMeshMixedDualB(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Create a computational mesh for the dual problem
TPZMultiphysicsCompMesh *createCompMeshMixedDualC(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Error estimation function for H1 solution using mixed solution as reference
REAL GoalEstimation(TPZMultiphysicsCompMesh* cmeshMixed, TPZCompMesh* cmesh, TPZVec<int> &refinementIndicator,int nthreads);

// Compute the error functional
REAL ComputeFunctional(TPZCompMesh* cmesh, int gthreads);

// Compute the error functional
REAL ComputeFunctionalB(TPZCompMesh* cmesh, int gthreads);

// Compute the error functional
REAL ComputeFunctionalC(TPZCompMesh* cmesh, int gthreads);

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

  int dim;
  if (testCase == 1 || testCase == 2) {
    gexact.fExact = TLaplaceExample1::ESinSin;
    dim = 2;
  }
  else if (testCase == 3) {
    gexact.fExact = TLaplaceExample1::ESphere;
    dim = 3;
  } else {
    std::cerr << "Test case not implemented!" << std::endl;
    return -1;
  }

  gperm = 1.0;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  // --- h-refinement loop ---

  int maxit = 6;
  int mixed_order = 1; // Polynomial order
  TPZVec<int> refinementIndicator;
  TPZVec<REAL> estimatedValues(maxit);
  TPZVec<REAL> exactValues(maxit);
  TPZVec<int> numberDofs(maxit);

  // Initial mesh
  TPZGeoMesh *gmesh = nullptr;
  if (testCase == 1 || testCase == 2) {
    gmesh = MeshingUtils::CreateGeoMesh2D(
      {4, 4, 4}, {0., 0., 0.}, {1., 1., 1.},
      {EDomain, EBoundary, EBoundary, EBoundary, EGoal});
  }
  else if (testCase == 3) {
    gmesh = MeshingUtils::ReadGeoMesh("mesh3D.msh");
  } else {
    std::cerr << "Test case not implemented!" << std::endl;
    return -1;
  }

  for (int iteration = 0; iteration < maxit; iteration++) {
    TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, mixed_order, false);

    TPZMultiphysicsCompMesh *cmeshDual = nullptr;
    if (testCase == 1) {
      cmeshDual = createCompMeshMixedDual(gmesh, mixed_order + 1, false);
    } else if (testCase == 2) {
      cmeshDual = createCompMeshMixedDualB(gmesh, mixed_order + 1, false);
    } else if (testCase == 3) {
      cmeshDual = createCompMeshMixedDual(gmesh, mixed_order + 1, false);
    } else {
      std::cerr << "Test case not implemented!" << std::endl;
      return -1;
    }

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
    numberDofs[iteration] = cmeshMixed->NEquations();

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

    if (shouldPlot) {
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
    }

    // --- Goal-oriented error estimation ---

    estimatedValues[iteration] = GoalEstimation(cmeshMixed, cmeshDual, refinementIndicator, gthreads);

    if (testCase == 1) {
      exactValues[iteration] = ComputeFunctional(cmeshMixed, gthreads);
    } else if (testCase == 2) {
      exactValues[iteration] = ComputeFunctionalB(cmeshMixed, gthreads);
    } else if (testCase == 3) {
      exactValues[iteration] = ComputeFunctional(cmeshMixed, gthreads);
    } else {
      std::cerr << "Test case not implemented!" << std::endl;
      return -1;
    }

    RefinementUtils::MeshSmoothing(gmesh, refinementIndicator);
    RefinementUtils::AdaptiveRefinement(gmesh, refinementIndicator);
    // RefinementUtils::UniformRefinement(gmesh);

    // --- Clean up ---

    delete cmeshDual;
    delete cmeshMixed;
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
  
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EBoundary, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);

  bcond = matDarcy->CreateBC(matDarcy, EGoal, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);

  if (testCase == 3) {
    bcond = matDarcy->CreateBC(matDarcy, ECylinder, 0, val1, val2);
    bcond->SetForcingFunctionBC(gexact.ExactSolution(), 6);
    hdivCreator->InsertMaterialObject(bcond);

    bcond = matDarcy->CreateBC(matDarcy, ECylinderBase, 0, val1, val2);
    bcond->SetForcingFunctionBC(gexact.ExactSolution(), 6);
    hdivCreator->InsertMaterialObject(bcond);
  }

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
  matDarcy->SetForcingFunction(ForcingFunctionDual, 6);
  hdivCreator->InsertMaterialObject(matDarcy);

  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 1.);
  
  val2[0] = 0.;
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EGoal, 0, val1, val2);
  hdivCreator->InsertMaterialObject(bcond);

  val2[0] = 0.;
  bcond = matDarcy->CreateBC(matDarcy, EBoundary, 0, val1, val2);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
  return cmesh;
}

TPZMultiphysicsCompMesh *createCompMeshMixedDualB(TPZGeoMesh *gmesh, int order, bool isCondensed) {
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
  bcond->SetForcingFunctionBC(DirichletFunctionDual, 6);
  hdivCreator->InsertMaterialObject(bcond);

  val2[0] = 0.;
  bcond = matDarcy->CreateBC(matDarcy, EBoundary, 0, val1, val2);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
  return cmesh;
}

REAL GoalEstimation(TPZMultiphysicsCompMesh* cmesh, TPZCompMesh* cmeshDual, TPZVec<int> &refinementIndicator, int nthreads) {

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

        // Invert the sign of dual solution
        for (auto &val : zh) val *= -1.0;
        for (auto &val : psih) val *= -1.0;
        for (auto &val : divpsih) val *= -1.0;

        // F contribution
        REAL FTerm = force[0]*zh[0];

        // Flux contribution
        REAL FluxTerm = 0.0;
        for (int d = 0; d < sigh.size(); ++d) {
          FluxTerm += (1./perm)*sigh[d]*psih[d];
        }

        // div contributions
        REAL divTermA = divsigh[0]*zh[0];
        REAL divTermB = -divpsih[0]*ph[0];

        goalError += (FTerm - FluxTerm - divTermA - divTermB) * weight;
      }

      // Boundary contribution to FTerm
      // int firstside = gel->NSides() - gel->Dimension() - 1;
      // int lastside = gel->NSides() - 1;

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
  
  // REAL maxError = *std::max_element(elementErrors.begin(), elementErrors.end(),
  //   [](REAL a, REAL b) { return std::abs(a) < std::abs(b); });
  // maxError = std::abs(maxError);

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

      // Check if element is condensed
      TPZCondensedCompEl *condEl = dynamic_cast<TPZCondensedCompEl *>(cel);
      if (condEl) {
        // If compel is condensed, load solution on the unconsensed compel
        condEl->LoadSolution();
        cel = condEl->ReferenceCompEl();
      }

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

        // Compute mixed solution ph and sigh
        TPZManVector<REAL,3> ph(gel->Dimension(),0.0);
        cel->Solution(ptInElement, 2, ph);


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

REAL ComputeFunctionalB(TPZCompMesh* cmesh, int nthreads) {
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

      // Check if element is condensed
      TPZCondensedCompEl *condEl = dynamic_cast<TPZCondensedCompEl *>(cel);
      if (condEl) {
        // If compel is condensed, load solution on the unconsensed compel
        condEl->LoadSolution();
        cel = condEl->ReferenceCompEl();
      }

      int matid = cel->Material()->Id();
      if (matid != EGoal) continue;
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
        TPZFMatrix<STATE> deriv;
        DirichletFunctionDual(x, force, deriv);

        // Compute mixed solution ph and sigh
        TPZManVector<REAL,3> ph(gel->Dimension(),0.0);
        TPZManVector<REAL,3> sigh(1,0.0);
        TPZManVector<REAL,1> divsigh(1,0.0);
        cel->Solution(ptInElement, 2, ph);
        cel->Solution(ptInElement, 18, sigh);

        // Compute exact solution pe
        TPZVec<STATE> pe(1);
        TPZFMatrix<REAL> dummyJac;
        TPZFMatrix<REAL> sige(3,1);
        gexact.ExactSolution()(x, pe, dummyJac);
        gexact.SigmaLoc(x, sige);

        // Multiply sige by normal (-1,0)
        sige(0,0) *= -1.0;

        elementContribution += force[0]*(sige(0,0)-sigh[0])*weight;
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