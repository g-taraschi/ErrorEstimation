#include <iostream>
#include <string>
#include <thread>

#include "DarcyFlow/TPZDarcyFlow.h"
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "TPZAnalyticSolution.h"
#include "TPZGeoMeshTools.h"
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
#include "TPZH1ApproxCreator.h"
#include "TPZHDivApproxCreator.h"
#include "pzcondensedcompel.h"
#include "pzstepsolver.h"

// Local includes
#include "MeshingUtils.h"

// ================
// Global variables
// ================

int gthreads = 18;

// Exact solution
TLaplaceExample1 gexact;

// Permeability
REAL gperm = 1.0;

// ===================
// Function prototypes
// ===================

// Creates a computational mesh for H1 approximation
TPZCompMesh *createCompMeshH1Old(TPZGeoMesh *gmesh, int order = 1);

// Creates a computational mesh for H1 approximation (with H1Creator)
TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1, bool isCondensed = false);

// Error estimation function for H1 solution using mixed solution as reference
REAL ErrorEstimation(TPZMultiphysicsCompMesh* cmeshMixed, TPZCompMesh* cmesh, TPZVec<REAL>& refinementIndicator, REAL rtol, int nthreads);

// Perform uniform refinement of the geometric mesh
void UniformRefinement(TPZGeoMesh *gmesh);

// Perform adaptive refinement of the geometric mesh
void AdaptiveRefinement(TPZGeoMesh *gmesh, TPZVec<REAL> refinementIndicator);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

// Initialize logger
#ifdef PZ_LOG
  TPZLogger::InitializePZLOG("logpz.txt");
#endif

  // --- Set a problem with analytic solution ---

  int dim = 3;
  gperm = 1.0;

  gexact.fExact = TLaplaceExample1::ESinSin;
  gexact.fDimension = dim;
  gexact.fTensorPerm = {{gperm, 0., 0.}, {0., gperm, 0.}, {0., 0., gperm}};

  bool shouldPlot = true; // Set to true for plotting the solutions

  // --- h-refinement loop ---

  REAL rtol = 0.5; // Relative refinement tolerance
  REAL estimatedError = 0.;
  std::ofstream hrefFile("hrefTest.txt");

  for (int h1_order = 1; h1_order < 4; h1_order++) {
    int mixed_order = 1;

    // Print polynomial orders for H1 and mixed approximations
    hrefFile << "\n\nOrder H1: " << h1_order
             << ", Order Mixed: " << mixed_order << std::endl;
   
    // Initial geometric mesh
    TPZGeoMesh *gmesh = nullptr;
    if(dim == 2) {
      gmesh = MeshingUtils::CreateGeoMesh2D({2, 2}, {0., 0.}, {1., 1.});
    } else if(dim == 3) {
      gmesh = MeshingUtils::CreateGeoMesh3D({2, 2, 2}, {0., 0., 0.}, {1., 1., 1.});
    }

    for (int iteration = 0; iteration < 5; iteration++) {
      // H1 and mixed computational meshes
      TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, mixed_order, true);
      TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, h1_order, true);

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
          const std::string plotfile = "mixed_plot";
          constexpr int vtkRes{0};
          TPZManVector<std::string, 2> fields = {"Flux", "Pressure"};
          auto vtk = TPZVTKGenerator(cmeshMixed, fields, plotfile, vtkRes);
          vtk.Do();
        }
      }

      // --- Error Estimation ---

      // TODO: Check if I'm picking the right error norms here

      // Exact error for H1 solution || K^(1/2) grad(p-ph) ||_0
      TPZVec<REAL> errorsH1(3, 0.);
      anH1.PostProcessError(errorsH1, false, std::cout);
      REAL errorH1 = errorsH1[2];

      // Exact error for mixed solution || -K^{-1/2} (sig - sigh) ||_0
      TPZVec<REAL> errorsMixed(5, 0.);
      anMixed.PostProcessError(errorsMixed, false, std::cout);
      REAL errorMixed = errorsMixed[1];

      // Compute estimated error and Prager-Synge gap
      TPZVec<REAL> refinementIndicator;
      estimatedError = ErrorEstimation(cmeshMixed, cmeshH1, refinementIndicator, rtol, gthreads);
      REAL PgSyGap = estimatedError*estimatedError - errorH1*errorH1 - errorMixed*errorMixed;
      
      // Refine the geometric mesh
      UniformRefinement(gmesh);
      // AdaptiveRefinement(gmesh, refinementIndicator);

      // Print results
      std::cout << "\nIteration " << iteration << ":\n"
                << "    Estimated Error = " << estimatedError
                << ", Actual error for H1 = " << errorH1
                << ", Effective index H1 = " << estimatedError / errorH1
                << "\n"
                << ", Actual error for Mixed = " << errorMixed
                << ", Effective index Mixed = " << estimatedError / errorMixed
                << ", Prager-Synge Gap = " << PgSyGap << std::endl;

      hrefFile << std::scientific << std::setprecision(3) << estimatedError
               << " & " << errorH1 << " & " << estimatedError / errorH1
               << " & " << errorMixed << " & "
               << estimatedError / errorMixed << " & " << PgSyGap << std::endl;

      // --- Clean up ---

      delete cmeshH1;
      delete cmeshMixed;
    }
    delete gmesh;
  }
}

// =========
// Functions
// =========

TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order, bool isCondensed) {
  TPZH1ApproxCreator h1Creator(gmesh);
  h1Creator.SetDefaultOrder(order);
  h1Creator.ProbType() = ProblemType::EDarcy;
  h1Creator.SetShouldCondense(isCondensed);

  // Insert material
  TPZDarcyFlow *mat = new TPZDarcyFlow(EDomain, gmesh->Dimension());
  mat->SetConstantPermeability(gperm);
  mat->SetForcingFunction(gexact.ForceFunc(), 4);
  mat->SetExactSol(gexact.ExactSolution(), 4);
  h1Creator.InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 1.);
  TPZFMatrix<REAL> val1(1, 1, 0.);
  val2[0] = 1.0;
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EBoundary, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  h1Creator.InsertMaterialObject(bcond);

  // Create the H1 computational mesh
  TPZCompMesh *cmesh = h1Creator.CreateClassicH1ApproximationSpace();
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
  TPZBndCondT<REAL> *bcond = matDarcy->CreateBC(matDarcy, EBoundary, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  hdivCreator->InsertMaterialObject(bcond);

  TPZMultiphysicsCompMesh *cmesh = hdivCreator->CreateApproximationSpace();
  return cmesh;
}

REAL ErrorEstimation(TPZMultiphysicsCompMesh* cmeshMixed, TPZCompMesh* cmesh, TPZVec<REAL>& refinementIndicator, REAL rtol, int nthreads) {

  nthreads++; // If nthreads = 0, we use 1 thread

  // Ensure references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();

  // Resizes refinementIndicator and fill with zeros
  int64_t ngel = cmesh->Reference()->NElements();
  refinementIndicator.Resize(ngel);
  refinementIndicator.Fill(0.0);

  int64_t ncel = cmeshMixed->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();
  TPZVec<REAL> elementErrors(ncel, 0.0);

  // Parallelization setup
  std::vector<std::thread> threads(nthreads);
  TPZManVector<REAL> partialErrors(nthreads, 0.0);

  auto worker = [&](int tid, int64_t start, int64_t end) {
    REAL localTotalError = 0.0;
    for (int64_t icel = start; icel < end; ++icel) {
      TPZCompEl *celMixed = elementvec_m[icel];

      // Check if mixed element is condensed
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
        // If compel is condensed, load solution on the unconsensed compel
        condEl->LoadSolution();
        celH1 = condEl->ReferenceCompEl();
      }

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
  for (int64_t i = 0; i < ncel; ++i) {
    if (elementErrors[i] > rtol*maxError) {
      int64_t igeo = cmeshMixed->Element(i)->Reference()->Index();
      refinementIndicator[igeo] = 1;
    }
  }

  return totalError;
}

void UniformRefinement(TPZGeoMesh *gmesh) {
  TPZCheckGeom checkgeom(gmesh);
  checkgeom.UniformRefine(1);
}

void AdaptiveRefinement(TPZGeoMesh *gmesh, TPZVec<REAL>& refinementIndicator) {
  for (int64_t iel = 0; iel < refinementIndicator.size(); ++iel) {
    if (refinementIndicator[iel] == 0) continue;
    TPZGeoEl *gel = gmesh->Element(iel);
    if (!gel) continue;
    if (gel->HasSubElement()) continue;
    TPZVec<TPZGeoEl *> pv;
    gel->Divide(pv);

    // Refine boundary
    int firstside = gel->FirstSide(2);
    int lastside = gel->FirstSide(3);
    for (int side = firstside; side < lastside; ++side) {
      TPZGeoElSide gelSide(gel, side);
      std::set<int> bcIds = {EBoundary};
      TPZGeoElSide neigh = gelSide.HasNeighbour(bcIds);
      if (neigh) {
        TPZGeoEl *neighGel = neigh.Element();
        if (neighGel->Dimension() != gmesh->Dimension()-1) continue;
        TPZVec<TPZGeoEl *> pv2;
        neighGel->Divide(pv2);
      }
    }
  }
}

TPZCompMesh *createCompMeshH1Old(TPZGeoMesh *gmesh, int order) {
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
  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EBoundary, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  cmesh->AutoBuild();
  return cmesh;
}