#include <iostream>
#include <thread>

// PZ includes
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "DarcyFlow/TPZDarcyFlow.h"
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
#include "TPZMaterialData.h"

// Local includes
#include "MeshingUtils.h"
#include "RefinementUtils.h"

// ================
// Global variables
// ================

int gthreads = 18;
bool shouldPlot = true;

// Exact solution
TLaplaceExample1 gexact;

// Permeability
REAL gperm = 1.0;

// Tolerance for error visualization
REAL gtol = 1e-3;
REAL rtol = 0.5;

// Forcing function for the dual problem
auto DirichletFunctionDual = [](const TPZVec<REAL> &pt, TPZVec<STATE> &result, TPZFMatrix<STATE> &deriv) {
  result.Resize(1); // Ensure proper size
  deriv.Resize(3, 1); // 2D gradient but needs 3 components

  REAL x = pt[0];
  REAL y = pt[1];

  result[0] = 0.; // Tests 1 and 2

  // Test 3
  // if (x > -1e-8 && x < 1e-8 && y <= 0.5 && y >= 0.2) {
  //   result[0] = 1.0/0.3; // Length of curve
  // }

  // Test 4
  // if (x <= 0.25 && y <= 0.25) {
  //   result[0] = 1.0 - 4.0 * (x + y);
  // }

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

  result[0] = 0.; // Test 3 and 4

  // Test 1
  result[0] = pow(x,10)*pow(y,10);

  // Test 2
  // if (x >= 0.75 && x <= 0.875 && y >= 0.75 && y <= 0.875) {
  //   result[0] = 1. / 0.015625; // divide by area of the square
  // }

  // Test 5
  // result[0] = 2.0*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y);
};

// ===================
// Function prototypes
// ===================

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Creates a computational mesh for H1 approximation
TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order = 1);

// Create a computational mesh for the dual problem
TPZMultiphysicsCompMesh *createCompMeshMixedDual(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Create an H1 computational mesh for the dual problem
TPZCompMesh *createCompMeshH1Dual(TPZGeoMesh *gmesh, int order = 1);

// Error estimation function for H1 solution using mixed solution as reference
REAL PragerSynge(TPZCompMesh* cmeshMixed, TPZCompMesh* cmeshH1, TPZVec<REAL> &elementErrors, bool isDual, int nthreads);

// Compute the error functional
REAL ComputeFunctional(TPZCompMesh* cmesh, int gthreads);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

// Initialize logger
#ifdef PZ_LOG
  TPZLogger::InitializePZLOG();
#endif

  // --- Solve darcy problem ---

  gexact.fExact = TLaplaceExample1::ESinSin;
  gperm = 1.0;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  // --- h-refinement loop ---

  int maxit = 6;
  int mixed_order = 1; // Polynomial order
  TPZVec<REAL> elementErrorsPrimal, elementErrorsDual, goalEstimator;
  TPZVec<REAL> exactFunctional(maxit);
  TPZVec<REAL> exactError(maxit);
  TPZVec<REAL> estimatedValuesPrimal(maxit);
  TPZVec<REAL> estimatedValuesDual(maxit);
  TPZVec<REAL> estimatedFunctional(maxit);
  TPZVec<int> numberDofs(maxit);

  TPZFMatrix<REAL> errors(maxit, 5);
  TPZVec<REAL> hs(maxit);

  // Initial mesh
  TPZGeoMesh *gmesh = MeshingUtils::CreateGeoMesh2D(
      {4, 4}, {0., 0.}, {1., 1.},
      {EDomain, EDirichlet, EDirichlet, EDirichlet, EDirichlet});

  for (int iteration = 0; iteration < maxit; iteration++) {
    TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, mixed_order, true);
    TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, mixed_order + 2);
    TPZMultiphysicsCompMesh *cmeshMixedDual = createCompMeshMixedDual(gmesh, mixed_order, true);
    TPZCompMesh *cmeshH1Dual = createCompMeshH1Dual(gmesh, mixed_order + 2);

    hs[iteration] = 1.0 / (pow(2, iteration)); // Fake mesh size for convergence study

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

    // Mixed solver dual
    TPZLinearAnalysis anMixedDual(cmeshMixedDual);
    TPZSSpStructMatrix<STATE> matMixedDual(cmeshMixedDual);
    matMixedDual.SetNumThreads(gthreads);
    anMixedDual.SetStructuralMatrix(matMixedDual);
    TPZStepSolver<STATE> stepMixedDual;
    stepMixedDual.SetDirect(ELDLt);
    anMixedDual.SetSolver(stepMixedDual);
    anMixedDual.Run();

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

    // H1 solver dual
    TPZLinearAnalysis anH1Dual(cmeshH1Dual);
    TPZSSpStructMatrix<STATE> matH1Dual(cmeshH1Dual);
    matH1Dual.SetNumThreads(gthreads);
    anH1Dual.SetStructuralMatrix(matH1Dual);
    TPZStepSolver<STATE> stepH1Dual;
    stepH1Dual.SetDirect(ECholesky);
    anH1Dual.SetSolver(stepH1Dual);
    anH1Dual.Run();

    // --- Plotting ---

    if (shouldPlot) {
      {
        const std::string plotfile = "mixed_plot";
        constexpr int vtkRes{0};
        TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
        auto vtk = TPZVTKGenerator(cmeshH1, fields, plotfile, vtkRes);
        vtk.Do();
      }

      {
        const std::string plotfile = "dual_plot";
        constexpr int vtkRes{0};
        TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
        auto vtk = TPZVTKGenerator(cmeshH1Dual, fields, plotfile, vtkRes);
        vtk.Do();
      }
    }

    // --- Goal-oriented error estimation ---

    estimatedValuesPrimal[iteration] = PragerSynge(cmeshMixed, cmeshH1, elementErrorsPrimal, false, gthreads);
    estimatedValuesDual[iteration] = PragerSynge(cmeshMixedDual, cmeshH1Dual, elementErrorsDual, true, gthreads);

    if (elementErrorsDual.NElements() != elementErrorsPrimal.NElements()) {
      DebugStop();
    }

    goalEstimator.Resize(elementErrorsPrimal.NElements());
    for (int64_t i = 0; i < elementErrorsPrimal.NElements(); i++) {
      goalEstimator[i] = sqrt(elementErrorsPrimal[i] * elementErrorsDual[i]);
    }

    estimatedFunctional[iteration] = 0.0;
    for (int64_t i = 0; i < goalEstimator.NElements(); i++) {
      estimatedFunctional[iteration] += goalEstimator[i];
    }

    TPZVec<REAL> errorsMixed(5, 0.);
    anMixed.PostProcessError(errorsMixed, false, std::cout);
    exactFunctional[iteration] = abs(ComputeFunctional(cmeshMixed, gthreads));
    exactError[iteration] = errorsMixed[1];

    // --- Refinement ---

    // RefinementUtils::MeshSmoothing(gmesh, refinementIndicator);
    // RefinementUtils::AdaptiveRefinement(gmesh, refinementIndicator);
    RefinementUtils::UniformRefinement(gmesh);

    // --- Clean up ---

    delete cmeshH1Dual;
    delete cmeshH1;
    delete cmeshMixedDual;
    delete cmeshMixed;
  }

  // --- Output results ---
  std::cout << "\nNumber of DoFs, Real values, Estimated Values, Eff. Index\n";
  for (int i = 0; i < estimatedFunctional.NElements(); i++) {
  std::cout << numberDofs[i] << " & "
        << std::scientific << std::setprecision(3) << exactError[i] << " & "
        << std::scientific << std::setprecision(3) << estimatedValuesPrimal[i] << " & "
        << std::fixed << std::setprecision(3) << estimatedValuesPrimal[i] / exactError[i] << " & "
        << std::scientific << std::setprecision(3) << exactFunctional[i] << " & "
        << std::scientific << std::setprecision(3) << estimatedFunctional[i] << " & "
        << std::fixed << std::setprecision(3) << estimatedFunctional[i] / exactFunctional[i] << " & "
        << std::fixed << std::setprecision(3) << estimatedValuesDual[i]*estimatedValuesPrimal[i] / exactFunctional[i] << "\n";
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
  matDarcy->SetForcingFunction(gexact.ForceFunc(), 6);
  matDarcy->SetExactSol(gexact.ExactSolution(), 6);
  hdivCreator->InsertMaterialObject(matDarcy);

  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 1.);
  
  val2[0] = 0.;
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EDirichlet, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 6);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
  return cmesh;
}

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
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EDirichlet, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 6);
  cmesh->InsertMaterialObject(bcond);

  cmesh->AutoBuild();

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
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EDirichlet, 0, val1, val2);
  bcond->SetForcingFunctionBC(DirichletFunctionDual, 6);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
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
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EDirichlet, 0, val1, val2);
  bcond->SetForcingFunctionBC(DirichletFunctionDual, 6);
  cmesh->InsertMaterialObject(bcond);

  cmesh->AutoBuild();

  return cmesh;
}

// Prager Synge error estimator for primal or dual problems
REAL PragerSynge(TPZCompMesh* cmeshMixed, TPZCompMesh* cmeshH1, TPZVec<REAL> &elementErrors, bool isDual, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread
  int dim = cmeshMixed->Dimension();

  // Ensure references point to H1 cmesh
  cmeshH1->Reference()->ResetReference();
  cmeshH1->LoadReferences();

  // Resizes elementErrors vector
  int64_t ngel = cmeshMixed->Reference()->NElements();
  elementErrors.Resize(ngel);
  elementErrors.Fill(0.0);

  int64_t ncel = cmeshMixed->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();

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

      // Check if H1 element is condensed
      condEl = dynamic_cast<TPZCondensedCompEl *>(celH1);
      if (condEl) {
        // If compel is condensed, load solution on the uncondensed compel
        condEl->LoadSolution();
        celH1 = condEl->ReferenceCompEl();
      }

      if (!celMixed || !celH1) continue;
      if (gel->HasSubElement()) continue;
      if (celH1->Material()->Id() != matid) DebugStop();

      REAL perm = gperm;
      REAL hk = MeshingUtils::ElementDiameter(gel); 
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
        weight *= detjac;

        TPZManVector<REAL, 3> x(3, 0.0);
        gel->X(ptInElement, x); // Real coordinates for x

        TPZManVector<REAL,3> force(1,0.0);
        if (isDual) {
          ForcingFunctionDual(x, force);
        } else {
          gexact.ForceFunc()(x, force);
        }

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
        for (int d = 0; d < dim; ++d) {
          REAL diff = termH1[d] - termHdiv[d];
          diffFlux += diff * diff;
        }

        // Balance contribution
        REAL diffBalance = (divFluxMixed[0] - force[0]) * (divFluxMixed[0] - force[0]);

        fluxError += diffFlux * weight;
        balanceError += diffBalance * weight;
      }

      REAL contribution = sqrt(fluxError) + (hk/(M_PI*sqrtPerm))*sqrt(balanceError);
      int64_t igeo = cmeshMixed->Element(icel)->Reference()->Index();
      elementErrors[igeo] = (contribution * contribution);

      localTotalError += elementErrors[igeo];
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
    if (isDual) {
      std::ofstream out_estimator("ErrorEstimationDual.vtk");
      TPZVTKGeoMesh::PrintCMeshVTK(cmeshH1, out_estimator, elementErrors,
                                   "EstimatedError");
    } else {
      std::ofstream out_estimator("ErrorEstimation.vtk");
      TPZVTKGeoMesh::PrintCMeshVTK(cmeshH1, out_estimator, elementErrors,
                                   "EstimatedError");
    }
  }

  return sqrt(totalError);
}

REAL ComputeFunctional(TPZCompMesh *cmesh, int nthreads) {
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

      TPZGeoEl *gel = cel->Reference();
      if (!gel) DebugStop();

      REAL elementContribution = 0.0;
      int matid = cel->Material()->Id();

      // Domain part
      if (matid == EDomain) {
        const TPZIntPoints *intrule = gel->CreateSideIntegrationRule(gel->NSides() - 1, 10);

        for (int ip = 0; ip < intrule->NPoints(); ++ip) {
          TPZManVector<REAL, 3> ptInElement(gel->Dimension());
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
          TPZManVector<REAL, 3> ph(gel->Dimension(), 0.0);
          cel->Solution(ptInElement, 2, ph);

          // Compute exact solution pe
          TPZVec<STATE> pe(1);
          TPZFMatrix<REAL> dummyJac;
          gexact.ExactSolution()(x, pe, dummyJac);

          elementContribution += force[0] * (pe[0] - ph[0]) * weight;
        }

      // Boundary part
      } else if (matid == EDirichlet) {
        // Get the normal vector
        TPZManVector<REAL, 3> qsi(gel->Dimension());
        gel->CenterPoint(gel->NSides() - 1, qsi);

        TPZFNMatrix<9, REAL> jac, axes, jacinv;
        REAL detjac;
        gel->Jacobian(qsi, jac, axes, detjac, jacinv);

        TPZManVector<REAL, 3> normal(3);
        normal[0] = axes(0,1);
        normal[1] = -axes(0,0);
        normal[2] = 0.0;

        // Normalize
        REAL norm = sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                         normal[2] * normal[2]);
        if (norm > 1e-12) {
          normal[0] /= norm;
          normal[1] /= norm;
          normal[2] /= norm;
        }

        // Set integration rule
        const TPZIntPoints *intrule = gel->CreateSideIntegrationRule(gel->NSides() - 1, 10);

        for (int ip = 0; ip < intrule->NPoints(); ++ip) {
          TPZManVector<REAL, 3> ptInElement(gel->Dimension());
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

          // Compute mixed solution ph and fluxh
          TPZManVector<REAL, 3> ph(gel->Dimension(), 0.0);
          TPZManVector<REAL, 3> fluxh(1, 0.0);
          cel->Solution(ptInElement, 18, fluxh);

          // Compute exact solution pe
          TPZFMatrix<REAL> sige(3, 1);
          gexact.SigmaLoc(x, sige);

          // Multiply sige by normal
          REAL fluxe = sige(0, 0) * normal[0] + sige(1, 0) * normal[1] + sige(2, 0) * normal[2];
          //REAL fluxe = -sige(0,0);

          elementContribution += force[0] * (fluxe - fluxh[0]) * weight;
        }
      }
      threadContribution += elementContribution;
    }
    threadContributions[tid] = threadContribution;
  };

  int64_t chunk = ncel / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    int64_t start = t * chunk;
    int64_t end = (t == nthreads - 1) ? ncel : (t + 1) * chunk;
    threads[t] = std::thread(worker, t, start, end);
  }
  for (auto &th : threads)
    th.join();

  REAL totalValue = 0.0;
  for (auto val : threadContributions)
    totalValue += val;

  return totalValue;
}