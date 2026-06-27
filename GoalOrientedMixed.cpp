#include <iostream>
#include <thread>

// NeoPZ includes
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "DarcyFlow/TPZDarcyFlow.h"
#include "TPZMultiphysicsCompMesh.h"
#include "TPZHDivApproxCreator.h"
#include "pzcondensedcompel.h"
#include "pzmultiphysicscompel.h"
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

// Local stuff
#include "Utils/MeshingUtils.h"
#include "Utils/RefinementUtils.h"

// ================
// Global variables
// ================

int gthreads = 11;
bool shouldPlot = true;

// Exact solution
TLaplaceExample1 gexact;

// Permeability
REAL gperm = 1.0;

// Material IDs for domain and boundaries
enum EnumMatIds {
  EDomain = 1,
  EDirichlet = 2,
  ENeumann = 3
};

enum EstimationType {
  EGoalEstimation = 1,
  EPragerSynge = 2
};

enum RefinementType {
  ERefineUniform = 1,
  ERefineAdaptive = 2
};

int gEstimationType = EGoalEstimation;
int gRefinementType = ERefineAdaptive;

// Dirichlet function for the dual problem
auto DirichletFunctionDual = [](const TPZVec<REAL> &pt, TPZVec<STATE> &result, TPZFMatrix<STATE> &deriv) {
  result.Resize(1); // Ensure proper size
  deriv.Resize(3, 1); // 2D gradient but needs 3 components

  REAL x = pt[0];
  REAL y = pt[1];
  REAL z = pt[2];

  result[0] = 0.;

//   REAL sig = 0.07;
//   result[0] = exp(-(x*x + y*y)/(2*sig*sig));

//   if (x < 0.125 && y < 0.125) {
//     result[0] = (1.0 - 8.0 * x) * (1.0 - 8.0 * y) * (1.0 - 8.0 * x) * (1.0 - 8.0 * y);
//   }

  if (x < 0.25 && y < 0.25) {
    result[0] = (1.0 - 4.0 * x) * (1.0 - 4.0 * y) * (1.0 - 4.0 * x) * (1.0 - 4.0 * y);
  }

//   if (x > 0.25 && x < 0.75 && y > 0.25 && y < 0.75) {
//     result[0] = 1.0;
//   }

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
  REAL z = pt[2];

  result[0] = 0.; // Test 2

  // Test 1
//   if ( 0.125 < x && x < 0.150 &&  0.125 < y && y < 0.150) {
//     result[0] = 1.0;
//   }
};

// ===================
// Function prototypes
// ===================

// Creates a computational mesh for H1 approximation
TPZCompMesh *CreateCompMeshH1(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *CreateCompMeshMixed(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData, bool isCondensed = false);

// Create a computational mesh for the dual problem
TPZMultiphysicsCompMesh *CreateCompMeshMixedDual(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData, bool isCondensed = false);

// First test for goal-oriented error estimation
REAL GoalEstimation(TPZCompMesh* cmeshMixed, TPZCompMesh* cmeshDual, TPZVec<REAL> &elementContributions, int nthreads = 0);

// Remove low-order connects from the dual mesh
void RemoveLowOrderConnects(TPZCompMesh *cmeshDual);

// Prager-Synge based error estimation
REAL PragerSynge(TPZCompMesh *cmeshMixed, TPZCompMesh *cmesh, TPZVec<REAL> &elementErrors, int nthreads = 0);

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

  // --- Set up ---

  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fDimension = 2;
  gperm = 1.0;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  // Initial mesh 2D
  TPZManVector<int, 5> matIds = {EDomain, EDirichlet, EDirichlet, EDirichlet, EDirichlet};
  // TPZGeoMesh *gmesh = MeshingUtils::CreateGeoMesh2D({4, 4}, {0., 0.}, {0.5, 0.5}, matIds);
  TPZGeoMesh *gmesh = MeshingUtils::ReadGeoMesh("well2D.msh", EDomain, EDirichlet, EDirichlet);

  // Initial mesh 3D
//   TPZManVector<int, 7> matIds = {EDomain, EDirichlet, EDirichlet, EDirichlet, EDirichlet, EDirichlet, EDirichlet};
//   TPZGeoMesh *gmesh = MeshingUtils::CreateGeoMesh3D({4, 4, 4}, {0., 0., 0.}, {1., 1., 1.}, matIds);


  // Vector with boundary condition data
  TPZVec<MeshingUtils::BoundaryData> bcData = {
    {EDirichlet, 0, 0.0}
  };

  // --- Adaptive refinement loop ---

  gEstimationType = EGoalEstimation;
  gRefinementType = ERefineAdaptive;
  int maxit = 5;
  REAL refTol = 0.2;
  REAL tol = 1e-8;
  int mixed_order = 1; // Polynomial order
  TPZVec<int> refinementIndicator;
  TPZVec<REAL> elementContributions;
  TPZVec<REAL> estimatedValues(maxit);
  TPZVec<REAL> exactValues(maxit);
  TPZVec<int> numberDofs(maxit);

  for (int iteration = 0; iteration < maxit; iteration++) {
    // Mixed mesh (main simulation)
    TPZMultiphysicsCompMesh *cmeshMixed = CreateCompMeshMixed(gmesh, mixed_order, bcData, true);

    // Aux mesh (dual or H1) - Used for error estimation
    TPZCompMesh *cmeshAux = nullptr;
    if (gEstimationType == EGoalEstimation) {
      cmeshAux = CreateCompMeshMixedDual(gmesh, mixed_order+1, bcData, true);
    } else if (gEstimationType == EPragerSynge) {
      cmeshAux = CreateCompMeshH1(gmesh, mixed_order + 2, bcData);
    } else {
      std::cerr << "Unknown estimation type!" << std::endl;
      DebugStop();
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

    // Aux solver
    TPZLinearAnalysis anAux(cmeshAux);
    TPZSSpStructMatrix<STATE> matAux(cmeshAux);
    matAux.SetNumThreads(gthreads);
    anAux.SetStructuralMatrix(matAux);
    TPZStepSolver<STATE> stepAux;
    stepAux.SetDirect(ELDLt);
    anAux.SetSolver(stepAux);
    anAux.Run();
    anAux.SetThreadsForError(gthreads);

    // --- Plotting ---

    if (shouldPlot) {
      {
        const std::string plotfile = "mixed_plot";
        constexpr int vtkRes{0};
        TPZManVector<std::string, 3> fields = {"Flux", "Pressure", "Derivative"};
        auto vtk = TPZVTKGenerator(cmeshMixed, fields, plotfile, vtkRes);
        vtk.Do();
      }

      {
        const std::string plotfile = "aux_plot";
        constexpr int vtkRes{0};
        TPZManVector<std::string, 3> fields = {"Flux", "Pressure", "Derivative"};
        auto vtk = TPZVTKGenerator(cmeshAux, fields, plotfile, vtkRes);
        vtk.Do();
      }
    }

    // --- Error estimation ---

    if (gEstimationType == EGoalEstimation) {
      // RemoveLowOrderConnects(cmeshAux); // test
      estimatedValues[iteration] = GoalEstimation(cmeshMixed, cmeshAux, elementContributions, gthreads);
    } else if (gEstimationType == EPragerSynge) {
      estimatedValues[iteration] = PragerSynge(cmeshMixed, cmeshAux, elementContributions, gthreads);
    } else {
      std::cerr << "Unknown estimation type!" << std::endl;
      DebugStop();
    }

    std::cout << "Iteration " << iteration << ": Estimated value = " << estimatedValues[iteration] << std::endl;

    if (gEstimationType == EGoalEstimation) {
      // Compute the exact value of the functional
      exactValues[iteration] = ComputeFunctional(cmeshMixed, gthreads);
    } else if (gEstimationType == EPragerSynge) {
      // Compute the exact value of the energy norm
    //   TPZVec<REAL> errorsMixed(5, 0.);
    //   anMixed.PostProcessError(errorsMixed, false, std::cout);
    //   exactValues[iteration] = errorsMixed[0];
      exactValues[iteration] = ComputeFunctional(cmeshMixed, gthreads);
    } else {
      std::cerr << "Unknown estimation type!" << std::endl;
      DebugStop();
    }

    if (gRefinementType == ERefineAdaptive) {
      RefinementUtils::MarkToRefineNew(elementContributions, refinementIndicator, refTol);
      RefinementUtils::MeshSmoothing(gmesh, refinementIndicator);
      RefinementUtils::RefineBoundary(gmesh, refinementIndicator, bcData);
      RefinementUtils::AdaptiveRefinement(gmesh, refinementIndicator);
    } else if (gRefinementType == ERefineUniform) {
      RefinementUtils::UniformRefinement(gmesh);
    } else {
      std::cerr << "Unknown refinement type!" << std::endl;
      DebugStop();
    }

    if (shouldPlot) {
      std::ofstream out("RefinedMesh.vtk");
      TPZVTKGeoMesh::PrintGMeshVTK(gmesh, out);
    }

    // --- Clean up ---

    delete cmeshAux;
    delete cmeshMixed;
  }

  // --- Output results ---

  std::cout << "\nNumber of DoFs, Real values, Estimated Values, Eff. Index\n";
  for (int i = 0; i < estimatedValues.NElements(); i++) {
  std::cout << numberDofs[i] << " & "
        << std::scientific << std::setprecision(3) << exactValues[i] << " & "
        << std::scientific << std::setprecision(3) << estimatedValues[i] << " & "
        << std::fixed << std::setprecision(3) << estimatedValues[i] / exactValues[i] << "\n";
  }

  std::cout << "\nNumber of DoFs, Real values, Estimated Values, Eff. Index\n";
  for (int i = 0; i < estimatedValues.NElements(); i++) {
  std::cout << "[" << numberDofs[i] << ", "
        << std::scientific << std::setprecision(3) << abs(exactValues[i]) << "]," << "\n";
  }
}

// =========
// Functions
// =========


TPZCompMesh *CreateCompMeshH1(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData) {
  TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
  cmesh->SetDimModel(gmesh->Dimension());
  cmesh->SetDefaultOrder(order);            // Polynomial order
  cmesh->SetAllCreateFunctionsContinuous(); // H1 Elements

  // Add materials (weak formulation)
  TPZDarcyFlow *mat = new TPZDarcyFlow(EDomain, gmesh->Dimension());
  mat->SetConstantPermeability(gperm); // Set constant permeability
  mat->SetForcingFunction(gexact.ForceFunc(), 5);
  mat->SetExactSol(gexact.ExactSolution(), 5);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 0.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  TPZBndCondT<REAL> *Bcond;
  for (const auto &bc : bcData) {
    Bcond = mat->CreateBC(mat, bc.matid, bc.type, val1, val2);
    Bcond->SetForcingFunctionBC(gexact.ExactSolution(), 5);
    cmesh->InsertMaterialObject(Bcond);
  }

  // Set up the computational mesh
  cmesh->AutoBuild();

  return cmesh;
}

TPZMultiphysicsCompMesh *CreateCompMeshMixed(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData, bool isCondensed) {
  TPZHDivApproxCreator hdivCreator(gmesh);
  hdivCreator.SetProbType(ProblemType::EDarcy);
  hdivCreator.HdivFamily() = HDivFamily::EHDivStandard;
  hdivCreator.SetDefaultOrder(order);
  hdivCreator.SetShouldCondense(isCondensed);

  TPZMixedDarcyFlow *reservoirMat = new TPZMixedDarcyFlow(EDomain, gmesh->Dimension());
  reservoirMat->SetExactSol(gexact.ExactSolution(), 5);
  reservoirMat->SetForcingFunction(gexact.ForceFunc(), 5);
  reservoirMat->SetConstantPermeability(1.0);
  hdivCreator.InsertMaterialObject(reservoirMat);

  // Boundary conditions ---
  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 0);
  TPZBndCondT<STATE> *BCond;
  for (const auto &bc : bcData) {
    BCond = reservoirMat->CreateBC(reservoirMat, bc.matid, bc.type, val1, val2);
    BCond->SetForcingFunctionBC(gexact.ExactSolution(), 5);
    hdivCreator.InsertMaterialObject(BCond);
  }

  TPZMultiphysicsCompMesh *cmesh = hdivCreator.CreateApproximationSpace();
  return cmesh;
}

TPZMultiphysicsCompMesh *CreateCompMeshMixedDual(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData, bool isCondensed) {
  TPZHDivApproxCreator hdivCreator(gmesh);
  hdivCreator.SetProbType(ProblemType::EDarcy);
  hdivCreator.HdivFamily() = HDivFamily::EHDivStandard;
  hdivCreator.SetDefaultOrder(order);
  hdivCreator.SetShouldCondense(isCondensed);

  TPZMixedDarcyFlow* reservoirMat = new TPZMixedDarcyFlow(EDomain, gmesh->Dimension());
  reservoirMat->SetConstantPermeability(gperm);
  reservoirMat->SetForcingFunction(ForcingFunctionDual, 5);
  hdivCreator.InsertMaterialObject(reservoirMat);

  // Boundary conditions ---
  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 0);
  TPZBndCondT<STATE> *BCond;

  // Essential boundary conditions
  for (const auto &bc : bcData) {
    if (bc.type == 1) { // Neumann
      val2[0] = 0.0;
      BCond = reservoirMat->CreateBC(reservoirMat, bc.matid, bc.type, val1, val2);
      hdivCreator.InsertMaterialObject(BCond);
    } else if (bc.type == 0) { // Dirichlet
      BCond = reservoirMat->CreateBC(reservoirMat, bc.matid, bc.type, val1, val2);
      BCond->SetForcingFunctionBC(DirichletFunctionDual, 5);
      hdivCreator.InsertMaterialObject(BCond);
    }
  }

  TPZMultiphysicsCompMesh *cmesh = hdivCreator.CreateApproximationSpace();
  return cmesh;
}

REAL GoalEstimation(TPZCompMesh* cmesh, TPZCompMesh* cmeshDual, TPZVec<REAL> &elementContributions, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread
  int dim = cmesh->Dimension();

  // Ensure references point to dual cmesh
  cmeshDual->Reference()->ResetReference();
  cmeshDual->LoadReferences();

  // Resizes elementContributions and fill with zeros
  int64_t ngel = cmesh->Reference()->NElements();
  elementContributions.Resize(ngel);
  elementContributions.Fill(0);

  int64_t ncel = cmesh->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmesh->ElementVec();
  TPZVec<REAL> elementContributionsAux(ncel, 0.0); // Auxiliary vector for plotting

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
      if (gel->HasSubElement()) continue;
      
      TPZCompEl *celDual = gel->Reference();

      // Check if element is condensed
      TPZCondensedCompEl *condElDual = dynamic_cast<TPZCondensedCompEl *>(celDual);
      if (condElDual) {
        // If compel is condensed, load solution on the unconsensed compel
        condElDual->LoadSolution();
        celDual = condElDual->ReferenceCompEl();
      }

      if (!celMixed || !celDual) continue;
      if (celDual->Material()->Id() != matid) DebugStop();

      REAL perm = gperm;
      REAL sqrtPerm = sqrt(perm);

      REAL goalError = 0.0;

      // Set integration rule
      const TPZIntPoints* intrule = nullptr;
      // intrule = &celMixed->GetIntegrationRule();
      intrule = gel->CreateSideIntegrationRule(gel->NSides() - 1, 10);

      // const TPZIntPoints &intruleMixed = celMixed->GetIntegrationRule();
      // const TPZIntPoints &intruleDual = celDual->GetIntegrationRule();
      // if (intruleMixed.NPoints() < intruleDual.NPoints()) {
      //   intrule = &intruleDual;
      // } else {
      //   intrule = &intruleMixed;
      // }

      for (int ip = 0; ip < intrule->NPoints(); ++ip) {
        TPZManVector<REAL,3> ptInElement(gel->Dimension());
        REAL weight, detjac;
        intrule->Point(ip, ptInElement, weight);
        TPZFNMatrix<9, REAL> jacobian, axes, jacinv;
        gel->Jacobian(ptInElement, jacobian, axes, detjac, jacinv);
        weight *= detjac; // fabs(detjac);

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
        REAL FTerm = -force[0]*zh[0];

        // Flux contribution
        REAL FluxTerm = 0.0;
        for (int d = 0; d < dim; ++d) {
          FluxTerm += (1./perm)*sigh[d]*psih[d];
        }

        // div contributions
        REAL divTermA = -divsigh[0]*zh[0];
        REAL divTermB = -divpsih[0]*ph[0];

        // Minus sign due to the definition of the dual problem
        goalError -= (FTerm - FluxTerm - divTermA - divTermB) * weight;
      }

      // Boundary contributions
      int firstSide = gel->FirstSide(dim-1);
      int lastSide = gel->FirstSide(dim);

      for (int side = firstSide; side < lastSide; ++side) {
        TPZGeoElSide gelSide(gel, side);
        std::set<int> bcIds = {EDirichlet};
        TPZGeoElSide neigh = gelSide.HasNeighbour(bcIds);
        if (neigh) {
          TPZGeoEl *gelB = neigh.Element();
          if (gelB->HasSubElement()) {
            std::cout << "Gel has sublement?" << gel->HasSubElement() << std::endl;
            DebugStop();
          };
          TPZCompEl *celDualB = gelB->Reference();
          if (!celDualB) DebugStop();
          if (gelB->HasSubElement()) DebugStop();

          // Check if element is condensed
          TPZCondensedCompEl *condEl = dynamic_cast<TPZCondensedCompEl *>(celDualB);
          if (condEl) {
            // If compel is condensed, load solution on the unconsensed compel
            condEl->LoadSolution();
            celDualB = condEl->ReferenceCompEl();
          }

          // const TPZIntPoints* intruleSide = &celDualB->GetIntegrationRule();
          TPZIntPoints* intruleSide = gelB->CreateSideIntegrationRule(gelB->NSides() - 1, 10);

          for (int ip = 0; ip < intruleSide->NPoints(); ++ip) {
            TPZManVector<REAL,3> ptInElement(gelB->Dimension());
            REAL weight, detjac;
            intruleSide->Point(ip, ptInElement, weight);
            TPZFNMatrix<9, REAL> jacobian, axes, jacinv;
            gelB->Jacobian(ptInElement, jacobian, axes, detjac, jacinv);
            weight *= detjac;

            TPZManVector<REAL, 3> x(3, 0.0);
            gelB->X(ptInElement, x); // Real coordinates for x

            // Compute dual solution psih
            TPZManVector<REAL, 3> psih(1, 0.0);
            celDualB->Solution(ptInElement, 18, psih);

            // Compute exact solution pe
            TPZVec<STATE> pe(1);
            TPZFMatrix<REAL> dummyJac;
            gexact.ExactSolution()(x, pe, dummyJac);

            // - sign due to the definition of the dual problem
            goalError -= -pe[0]*psih[0]*weight;
          }
        }
      }

      int64_t igeo = cmesh->Element(icel)->Reference()->Index();
      elementContributions[igeo] = std::abs(goalError);
      elementContributionsAux[icel] = std::abs(goalError); // For plotting
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
    TPZVTKGeoMesh::PrintCMeshVTK(cmesh, out_estimator, elementContributionsAux,
                                 "EstimatedError");
  }

  return totalError;
}

REAL PragerSynge(TPZCompMesh* cmeshMixed, TPZCompMesh* cmesh, TPZVec<REAL>& elementErrors, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread

  // Ensure references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();

  int64_t ncel = cmeshMixed->NElements();
  int64_t ngel = cmeshMixed->Reference()->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();
  elementErrors.Resize(ngel); // Ensure proper size
  elementErrors.Fill(0.0); 

  TPZVec<REAL> elementErrorsAux(ncel, 0.0); // Aux vector for plotting

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

      REAL hk = MeshingUtils::ElementDiameter(gel);
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
        gexact.ForceFunc()(x, force);

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
        } else {
          DebugStop(); // celMixed should be multiphysics
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

      if (false) {
        std::ofstream errorlog("error_estimation.txt", std::ios_base::app);
        if (errorlog.is_open()) {
          errorlog << "Element " << icel << " - MatId " << matid
            << ": Flux error = " << fluxError
            << ", Balance error = " << balanceError << std::endl;
          errorlog.close();
        }
      }

      REAL contribution = sqrt(fluxError) + (hk/(M_PI*sqrt(perm)))*sqrt(balanceError);
      int64_t igeo = cmeshMixed->Element(icel)->Reference()->Index();
      elementErrors[igeo] = (contribution * contribution);
      elementErrorsAux[icel] = (contribution * contribution); // For plotting

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
  
  totalError = sqrt(totalError);

  // VTK output
  {
    std::ofstream out_estimator("EstimatedError.vtk");
    TPZVTKGeoMesh::PrintCMeshVTK(cmeshMixed, out_estimator, elementErrorsAux, "EstimatedError");
  }

  return totalError;
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
        // const TPZIntPoints *intrule = &cel->GetIntegrationRule();

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
        // Get the normal vector of the boundary geometric element.
        TPZManVector<REAL, 3> qsi(gel->Dimension(), 0.0);
        gel->CenterPoint(gel->NSides() - 1, qsi);

        TPZFNMatrix<9, REAL> jac, axes, jacinv;
        REAL detjac;
        gel->Jacobian(qsi, jac, axes, detjac, jacinv);

        TPZManVector<REAL, 3> normal(3, 0.0);
        if (gel->Dimension() == 1) {
          normal[0] = axes(0, 1);
          normal[1] = -axes(0, 0);
          normal[2] = 0.0;
        } else if (gel->Dimension() == 2) {
          // TODO: check
          normal[0] = axes(0, 1) * axes(1, 2) - axes(0, 2) * axes(1, 1);
          normal[1] = axes(0, 2) * axes(1, 0) - axes(0, 0) * axes(1, 2);
          normal[2] = axes(0, 0) * axes(1, 1) - axes(0, 1) * axes(1, 0);
        } else {
          DebugStop();
        }

        REAL norm = sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                         normal[2] * normal[2]);
        if (norm > 1e-12) {
          normal[0] /= norm;
          normal[1] /= norm;
          normal[2] /= norm;
        }

        // Set integration rule
        const TPZIntPoints *intrule = gel->CreateSideIntegrationRule(gel->NSides() - 1, 10);
        // const TPZIntPoints *intrule = &cel->GetIntegrationRule();

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

void RemoveLowOrderConnects(TPZCompMesh *cmesh) {
    cmesh->Reference()->ResetReference();
    cmesh->LoadReferences();
    int dimMesh = cmesh->Dimension();
    TPZFMatrix<STATE> &cmesh_sol = cmesh->Solution();

    for (auto cel : cmesh->ElementVec()) {

        // Check if element is condensed
        TPZCondensedCompEl *condEl = dynamic_cast<TPZCondensedCompEl *>(cel);
        if (condEl) {
          // If compel is condensed, load solution on the unconsensed compel
          condEl->LoadSolution();
          cel = condEl->ReferenceCompEl();
        }

        TPZGeoEl *gel = cel->Reference();
        if (!gel) DebugStop(); // gel should exist
        int dim = gel->Dimension();
        if (dim != dimMesh) continue; // Only process elements of the same dimension as the mesh
        int NCorners = gel->NCornerNodes();
        int NFaces = gel->FirstSide(dim) - gel->FirstSide(dim-1);

        TPZMultiphysicsElement *multiCel = dynamic_cast<TPZMultiphysicsElement *>(cel);
        if (!multiCel) DebugStop(); // Ensure cel is a multiphysics element

        // Get atomic compels
        TPZCompEl *hdivEl = multiCel->Element(0); // H(div) element
        TPZCompEl *h1El = multiCel->Element(1);   // H1 element

        if (!hdivEl || !h1El) DebugStop(); // Both elements should exist

        // Remove low-order H1 connects
        for (int i = 0; i < NCorners; ++i) {
          int64_t seq = h1El->Connect(i).SequenceNumber();

          auto firstEq = cmesh->Block().Position(seq);

          int64_t blockSize = cmesh->Block().Size(seq);

          for (int64_t eq = firstEq; eq < firstEq + blockSize; eq++) {
            cmesh_sol.PutVal(eq, 0, 0.0); // Set low-order H1 connects to zero
          }
        }

        // Remove low-order H(div) connects
        for (int i = 0; i < NFaces; ++i) {
          int64_t seq = hdivEl->Connect(i).SequenceNumber();

          auto firstEq = cmesh->Block().Position(seq);
          int64_t blockSize = cmesh->Block().Size(seq);

          // Assuming 2D element at first
          int lowOrderFunctions = 2; 

          for (int64_t eq = firstEq; eq < firstEq + lowOrderFunctions; eq++) {
            cmesh_sol.PutVal(eq, 0, 0.0); // Set low-order H(div) connects to zero
          }
        }
    }

    cmesh->TransferMultiphysicsSolution();
}