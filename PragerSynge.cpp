#include <fstream>
#include <iostream>

// NeoPZ includes
#include "DarcyFlow/TPZDarcyFlow.h"
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include <TPZHDivApproxCreator.h>
#include "TPZAnalyticSolution.h"
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
#include <pzcondensedcompel.h>

// Local stuff
#include "Utils/MeshingUtils.h"
#include "Utils/RefinementUtils.h"

// ================
// Global variables
// ================

// Exact solution
TLaplaceExample1 gexact;
bool shouldPlot = true;

// Permeability
REAL gperm = 1.0;

// Number of threads for parallel computations
const int gnthreads = 8;

// Material IDs for domain and boundaries
enum EnumMatIds {
  EDomain = 1,
  EDirichlet = 2,
  ENeumann = 3
};

// ===================
// Function prototypes
// ===================

// Creates a computational mesh for H1 approximation
TPZCompMesh *CreateCompMeshH1(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *CreateCompMeshMixed(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData, bool condensed = false);

// Error estimation function for H1 solution using mixed solution as reference
REAL PragerSynge(TPZCompMesh *cmesh, TPZMultiphysicsCompMesh *cmeshMixed, TPZVec<REAL> &elementErrors, int nthreads = 0);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

  gperm = 1.0; // Constant permeability
  int order = 1; // Polynomial order
  int maxIterations = 7; // Number of refinement iterations
  REAL errorTol = 0.9; // Error tolerance for refinement

  // Set a problem with analytic solution
  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fTensorPerm = {{gperm, 0., 0.},{0., gperm, 0.},{0., 0., gperm}};
  gexact.fDimension = 3;

  // TPZManVector<int, 5> matIDs = {EDomain, EDirichlet, EDirichlet, EDirichlet, EDirichlet};
  TPZManVector<int, 7> matIDs = {EDomain, EDirichlet, EDirichlet, EDirichlet, EDirichlet, EDirichlet, EDirichlet};

  // Vector with boundary condition data
  TPZVec<MeshingUtils::BoundaryData> bcData = {
    {EDirichlet, 0, 0.0}
  };

  // Initial mesh
  // TPZGeoMesh *gmesh = MeshingUtils::CreateGeoMesh2D({4, 4}, {0., 0.}, {1., 1.}, matIDs);
  TPZGeoMesh *gmesh = MeshingUtils::CreateGeoMesh3D({4, 4, 4}, {0., 0., 0.}, {1., 1., 1.}, matIDs);

  for (int iteration = 0; iteration < maxIterations; iteration++) {
    TPZMultiphysicsCompMesh *cmeshMixed = CreateCompMeshMixed(gmesh, order, bcData, true);
    TPZCompMesh *cmeshH1 = CreateCompMeshH1(gmesh, order+2, bcData);

    // Mixed solver
    TPZLinearAnalysis anMixed(cmeshMixed);
    TPZSSpStructMatrix<STATE> matMixed(cmeshMixed);
    matMixed.SetNumThreads(gnthreads);
    anMixed.SetStructuralMatrix(matMixed);
    TPZStepSolver<STATE> stepMixed;
    stepMixed.SetDirect(ELDLt);
    anMixed.SetSolver(stepMixed);
    anMixed.Run();

    // H1 solver
    TPZLinearAnalysis anH1(cmeshH1);
    TPZSSpStructMatrix<STATE> matH1(cmeshH1);
    matH1.SetNumThreads(gnthreads);
    anH1.SetStructuralMatrix(matH1);
    TPZStepSolver<STATE> stepH1;
    stepH1.SetDirect(ECholesky);
    anH1.SetSolver(stepH1);
    anH1.Run();

    // ---- Plotting ---

    if (shouldPlot) {
      const std::string plotfile = "PG_H1Plot";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshH1, fields, plotfile, vtkRes);
      vtk.Do();
    }

    if (shouldPlot) {
      const std::string plotfile = "PG_MixedPlot";
      constexpr int vtkRes{0};
      TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
      auto vtk = TPZVTKGenerator(cmeshMixed, fields, plotfile, vtkRes);
      vtk.Do();
    }

    // --- Error Estimation ---

    TPZVec<REAL> errorsEstimated;
    REAL EstimatedError = PragerSynge(cmeshH1, cmeshMixed, errorsEstimated, gnthreads);

    // TODO: Print real and estimated errors

    // --- Refine mesh ---
    TPZVec<int> refinementIndicator(gmesh->NElements(), 0);
    RefinementUtils::MarkToRefine(errorsEstimated, refinementIndicator, errorTol);
    RefinementUtils::MeshSmoothing(gmesh, refinementIndicator);
    RefinementUtils::RefineBoundary(gmesh, refinementIndicator, bcData);
    RefinementUtils::AdaptiveRefinement(gmesh, refinementIndicator);

    if (shouldPlot) {
      std::ofstream out("RefinedMesh.vtk");
      TPZVTKGeoMesh::PrintGMeshVTK(gmesh, out);
    }

    // --- Clean up ---

    // Remove dependencies before deleting H1 cmesh
    int ncon = cmeshH1->NConnects();
    for (int i = 0; i < ncon; ++i) {
      cmeshH1->ConnectVec()[i].RemoveDepend();
    }
    delete cmeshH1, cmeshMixed;
  }
  delete gmesh;
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
  mat->SetForcingFunction(gexact.ForceFunc(), 3);
  mat->SetExactSol(gexact.ExactSolution(), 3);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 0.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  TPZBndCondT<REAL> *Bcond;
  for (const auto &bc : bcData) {
    Bcond = mat->CreateBC(mat, bc.matid, bc.type, val1, val2);
    Bcond->SetForcingFunctionBC(gexact.ExactSolution(), 3);
    cmesh->InsertMaterialObject(Bcond);
  }

  // Set up the computational mesh
  cmesh->AutoBuild();

  return cmesh;
}

TPZMultiphysicsCompMesh *CreateCompMeshMixed(TPZGeoMesh *gmesh, int order, TPZVec<MeshingUtils::BoundaryData> &bcData, bool isCondensed) {
  TPZHDivApproxCreator hdivCreator(gmesh);
  hdivCreator.ProbType() = ProblemType::EDarcy;
  hdivCreator.HdivFamily() = HDivFamily::EHDivStandard;
  hdivCreator.SetDefaultOrder(order);
  hdivCreator.SetShouldCondense(isCondensed);

  TPZMixedDarcyFlow *reservoirMat = new TPZMixedDarcyFlow(EDomain, gmesh->Dimension());
  reservoirMat->SetExactSol(gexact.ExactSolution(), 3);
  reservoirMat->SetForcingFunction(gexact.ForceFunc(), 3);
  reservoirMat->SetConstantPermeability(1.0);
  hdivCreator.InsertMaterialObject(reservoirMat);

  // Boundary conditions ---
  TPZFMatrix<STATE> val1(1, 1, 0.);
  TPZManVector<STATE> val2(1, 0);
  TPZBndCondT<STATE> *BCond;
  for (const auto &bc : bcData) {
    BCond = reservoirMat->CreateBC(reservoirMat, bc.matid, bc.type, val1, val2);
    BCond->SetForcingFunctionBC(gexact.ExactSolution(), 1);
    hdivCreator.InsertMaterialObject(BCond);
  }

  TPZMultiphysicsCompMesh *cmesh = hdivCreator.CreateApproximationSpace();
  return cmesh;
}

REAL PragerSynge(TPZCompMesh* cmesh, TPZMultiphysicsCompMesh* cmeshMixed, TPZVec<REAL>& elementErrors, int nthreads) {

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