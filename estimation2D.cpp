#include "DarcyFlow/TPZDarcyFlow.h"
#include "DarcyFlow/TPZMixedDarcyFlow.h"
#include "TPZAnalyticSolution.h"
#include "TPZGenGrid2D.h"
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
#include <fstream>
#include <iostream>

// ================
// Global variables
// ================

// Exact solution
TLaplaceExample1 gexact;

// Permeability
REAL gperm = 1.0;

// Material IDs for domain and boundaries
enum EnumMatIds {
  EMatId = 1,
  EBottom = 2,
  ERight = 3,
  ETop = 4,
  ELeft = 5,
};

// ===================
// Function prototypes
// ===================

// Creates a geometric mesh using TPZGenGrid2D
TPZGeoMesh *createGeoMesh(const TPZManVector<int, 2> &nelDiv,
                          const TPZManVector<REAL, 2> &minX,
                          const TPZManVector<REAL, 2> &maxX);

// Computes the diameter of a geometric element
REAL ElementDiameter(TPZGeoEl *gel);

// Computes the diameter of a mesh
REAL MeshDiameter(TPZGeoMesh *gmesh);

// Creates a computational mesh for H1 approximation
TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order = 1);

// Creates a computational mesh for mixed approximation
TPZMultiphysicsCompMesh *createCompMeshMixed(TPZGeoMesh *gmesh, int order = 1);

// Error estimation function for H1 solution using mixed solution as reference
REAL ErrorEstimationH1(TPZCompMesh *cmesh, TPZMultiphysicsCompMesh *cmeshMixed);

// Error estimation function for mixed solution using H1 solution as reference
REAL ErrorEstimationMixed(TPZCompMesh *cmesh,
                          TPZMultiphysicsCompMesh *cmeshMixed, int order);

// =============
// Main function
// =============

int main(int argc, char *const argv[]) {

// --- Set up ---

// Initialize logger
#ifdef PZ_LOG
  TPZLogger::InitializePZLOG("logpz.txt");
#endif

  // Initialize uniform refinements for 1D and 2D elements
  gRefDBase.InitializeUniformRefPattern(EOned);
  gRefDBase.InitializeUniformRefPattern(EQuadrilateral);
  gRefDBase.InitializeUniformRefPattern(ETriangle);

  const int nthreads = 16;

  gperm = 1.0;

  // --- Solve darcy problem ---

  int order = 1; // Polynomial order

  // Set a problem with analytic solution
  gexact.fExact = TLaplaceExample1::ESinSin;
  // gexact.fExact = TLaplaceExample1::EHarmonic;
  gexact.fTensorPerm = {{gperm, 0., 0.},{0., gperm, 0.},{0., 0., gperm}};

  // --- Uniform h-refinement ---

  std::cout << "\n===== Starting uniform h-refinement test =====" << std::endl;
  TPZGeoMesh *gmesh = createGeoMesh({2, 2}, {0., 0.}, {1., 1.});

  for (int iteration = 0; iteration < 6; iteration++) {
    TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, order);
    TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, order+2);

    // Mixed solver
    TPZLinearAnalysis anMixed(cmeshMixed);
    TPZSSpStructMatrix<STATE> matMixed(cmeshMixed);
    matMixed.SetNumThreads(nthreads);
    anMixed.SetStructuralMatrix(matMixed);
    TPZStepSolver<STATE> stepMixed;
    stepMixed.SetDirect(ELDLt);
    anMixed.SetSolver(stepMixed);
    anMixed.Run();

    // H1 solver
    TPZLinearAnalysis anH1(cmeshH1);
    TPZSSpStructMatrix<STATE> matH1(cmeshH1);
    matH1.SetNumThreads(nthreads);
    anH1.SetStructuralMatrix(matH1);
    TPZStepSolver<STATE> stepH1;
    stepH1.SetDirect(ECholesky);
    anH1.SetSolver(stepH1);
    anH1.Run();

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

    REAL EstimatedErrorH1 = ErrorEstimationH1(cmeshH1, cmeshMixed);
    // REAL EstimatedErrorMixed = ErrorEstimationMixed(cmeshH1, cmeshMixed, order);

    // Exact error for H1 solution
    TPZVec<REAL> errorsH1(3, 0.);
    anH1.PostProcessError(errorsH1, false, std::cout);

    // Exact error for mixed solution
    TPZVec<REAL> errorsMixed(5, 0.);
    anMixed.PostProcessError(errorsMixed, false, std::cout);
    errorsMixed[1] = (1./sqrt(gperm)) * errorsMixed[1];

    // Print results
    std::cout << "\nIteration " << iteration << ":\n"
              << "    Estimated Error for H1 = " << EstimatedErrorH1
              << ", Actual error for H1 = " << errorsH1[0]
              << ", Effective index H1 = " << EstimatedErrorH1/errorsH1[0] << "\n"
              << "    Estimated Error for Mixed = " << EstimatedErrorH1
              << ", Actual error for Mixed = " << errorsMixed[1]
              << ", Effective index Mixed = " << EstimatedErrorH1/errorsMixed[1]
              << std::endl;

    std::ofstream hrefFile("hrefTest.txt", std::ios::app);
    hrefFile << std::scientific << std::setprecision(3)
            << EstimatedErrorH1 << " & " << errorsH1[0] << " & "
            << EstimatedErrorH1 / errorsH1[0] << " & "
            << EstimatedErrorH1 << " & " << errorsMixed[1] << " & "
            << EstimatedErrorH1 / errorsMixed[1] << std::endl;

    // --- Refine mesh ---

    // Uniform refinement
    TPZCheckGeom checkgeom(gmesh);
    checkgeom.UniformRefine(1);

    // --- Clean up ---

    delete cmeshH1, cmeshMixed;
  }
  delete gmesh;

  // --- Uniform p-refinement ---

  std::cout << "\n===== Starting uniform p-refinement test =====" << std::endl;
  gmesh = createGeoMesh({4, 4}, {0., 0.}, {1., 1.}); 
  order = 1;

  for (int iteration = 0; iteration < 0; iteration++) {
    TPZMultiphysicsCompMesh *cmeshMixed = createCompMeshMixed(gmesh, order);
    TPZCompMesh *cmeshH1 = createCompMeshH1(gmesh, order);

    // Mixed solver
    TPZLinearAnalysis anMixed(cmeshMixed);
    TPZSSpStructMatrix<STATE> matMixed(cmeshMixed);
    matMixed.SetNumThreads(nthreads);
    anMixed.SetStructuralMatrix(matMixed);
    TPZStepSolver<STATE> stepMixed;
    stepMixed.SetDirect(ELDLt);
    anMixed.SetSolver(stepMixed);
    anMixed.Run();

    // H1 solver
    TPZLinearAnalysis anH1(cmeshH1);
    TPZSSpStructMatrix<STATE> matH1(cmeshH1);
    matH1.SetNumThreads(nthreads);
    anH1.SetStructuralMatrix(matH1);
    TPZStepSolver<STATE> stepH1;
    stepH1.SetDirect(ECholesky);
    anH1.SetSolver(stepH1);
    anH1.Run();

    // --- Error Estimation ---

    REAL EstimatedErrorH1 = ErrorEstimationH1(cmeshH1, cmeshMixed);
    REAL EstimatedErrorMixed = ErrorEstimationMixed(cmeshH1, cmeshMixed, order);

    // Exact error for H1 solution
    TPZVec<REAL> errorsH1(3, 0.);
    anH1.PostProcessError(errorsH1, false, std::cout);

    // Exact error for mixed solution
    TPZVec<REAL> errorsMixed(5, 0.);
    anMixed.PostProcessError(errorsMixed, false, std::cout);
    errorsMixed[3] = sqrt(gperm) * errorsMixed[3];

    // Print results
    std::cout << "\nIteration " << iteration << ":\n"
              << "    Estimated Error for H1 = " << EstimatedErrorH1
              << ", Actual error for H1 = " << errorsH1[0]
              << ", Effective index H1 = " << EstimatedErrorH1/errorsH1[0] << "\n"
              << "    Estimated Error for Mixed = " << EstimatedErrorMixed
              << ", Actual error for Mixed = " << errorsMixed[3]
              << ", Effective index Mixed = " << EstimatedErrorMixed/errorsMixed[3]
              << std::endl;

    std::ofstream prefFile("prefTest.txt", std::ios::app);
    prefFile << std::scientific << std::setprecision(3)
            << cmeshH1->NEquations() << " & "
            << EstimatedErrorH1 << " & " << errorsH1[0] << " & "
            << EstimatedErrorH1 / errorsH1[0] << " & "
            << cmeshMixed->NEquations() << " & "
            << EstimatedErrorMixed << " & " << errorsMixed[3] << " & "
            << EstimatedErrorMixed / errorsMixed[3] << std::endl;

    order++; // Increase polynomial order
    delete cmeshH1, cmeshMixed;
  }
  delete gmesh;
}

// =========
// Functions
// =========

TPZGeoMesh *createGeoMesh(const TPZManVector<int, 2> &nelDiv,
                          const TPZManVector<REAL, 2> &minX,
                          const TPZManVector<REAL, 2> &maxX) {

  TPZGeoMesh *gmesh = new TPZGeoMesh;
  TPZGenGrid2D generator(nelDiv, minX, maxX);
  generator.SetElementType(MMeshType::EQuadrilateral);
  //generator.SetElementType(MMeshType::ETriangular);
  generator.Read(gmesh, EMatId);
  generator.SetBC(gmesh, 4, EBottom);
  generator.SetBC(gmesh, 5, ERight);
  generator.SetBC(gmesh, 6, ETop);
  generator.SetBC(gmesh, 7, ELeft);

  return gmesh;
}

TPZCompMesh *createCompMeshH1(TPZGeoMesh *gmesh, int order) {
  TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
  cmesh->SetDimModel(gmesh->Dimension());
  cmesh->SetDefaultOrder(order);            // Polynomial order
  cmesh->SetAllCreateFunctionsContinuous(); // H1 Elements

  // Add materials (weak formulation)
  TPZDarcyFlow *mat = new TPZDarcyFlow(EMatId, gmesh->Dimension());
  mat->SetConstantPermeability(gperm); // Set constant permeability
  mat->SetForcingFunction(gexact.ForceFunc(), 4);
  mat->SetExactSol(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(mat);

  // Add boundary conditions
  TPZManVector<REAL, 1> val2(1, 3.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EBottom, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ERight, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ETop, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ELeft, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
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
  TPZNullMaterial<STATE> *mat = new TPZNullMaterial(EMatId, gmesh->Dimension());
  cmeshFlux->InsertMaterialObject(mat);

  // Create boundary conditions
  TPZManVector<REAL, 1> val2(1, 3.); // Part that goes to the RHS vector
  TPZFMatrix<REAL> val1(1, 1, 0.);   // Part that goes to the Stiffnes matrix

  TPZBndCondT<REAL> *bcond = mat->CreateBC(mat, EBottom, 0, val1, val2);
  cmeshFlux->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ERight, 0, val1, val2);
  cmeshFlux->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ETop, 0, val1, val2);
  cmeshFlux->InsertMaterialObject(bcond);

  bcond = mat->CreateBC(mat, ELeft, 0, val1, val2);
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
      new TPZMixedDarcyFlow(EMatId, gmesh->Dimension());
  matDarcy->SetConstantPermeability(gperm);
  matDarcy->SetForcingFunction(gexact.ForceFunc(), 4);
  matDarcy->SetExactSol(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(matDarcy);

  // Create, set and add boundary conditions
  bcond = matDarcy->CreateBC(matDarcy, EBottom, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  bcond = matDarcy->CreateBC(matDarcy, ERight, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  bcond = matDarcy->CreateBC(matDarcy, ETop, 0, val1, val2);
  bcond->SetForcingFunctionBC(gexact.ExactSolution(), 4);
  cmesh->InsertMaterialObject(bcond);

  bcond = matDarcy->CreateBC(matDarcy, ELeft, 0, val1, val2);
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

REAL MeshDiameter(TPZGeoMesh *gmesh) {
  REAL h = 0.;
  int64_t nel = gmesh->NElements();
  for (int64_t el = 0; el < nel; ++el) {
    TPZGeoEl *gel = gmesh->Element(el);
    if (gel && gel->Dimension() == gmesh->Dimension() &&
        !gel->HasSubElement()) {
      REAL elSize = ElementDiameter(gel);
      if (elSize > h)
        h = elSize;
    }
  }
  return h;
}

REAL ErrorEstimationH1(TPZCompMesh *cmesh,
                        TPZMultiphysicsCompMesh *cmeshMixed) {
  // Ensures references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();

  int64_t ncel = cmeshMixed->NElements();
  int dim = cmeshMixed->Dimension();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();

  // Initialize error storage
  REAL totalError = 0.0;
  TPZIntPoints *intrule = nullptr; // Integration rule for element computations

  for (int64_t cel = 0; cel < ncel; ++cel) {

    TPZCompEl *celMixed = elementvec_m[cel];
    TPZGeoEl *gel = celMixed->Reference();
    TPZCompEl *celH1 = gel->Reference();

    if (!gel || gel->Dimension() != dim || gel->HasSubElement())
      continue;
    if (!celMixed || !celH1)
      continue;

    REAL hk = ElementDiameter(gel); // Element size

    // Define an integration rule
    const TPZIntPoints &intruleMixed = celMixed->GetIntegrationRule();
    const TPZIntPoints &intruleH1 = celH1->GetIntegrationRule();

    if (intruleMixed.NPoints() < intruleH1.NPoints()) {
      intrule = intruleH1.Clone();
    } else {
      intrule = intruleMixed.Clone();
    }

    // intrule = gel->CreateSideIntegrationRule(gel->NSides() - 1, 20);
    // if (!intrule)
    //   continue;

    REAL fluxError = 0.0;
    REAL balanceError = 0.0;
    REAL elementError = 0.0;
    int npts = intrule->NPoints();

    for (int ip = 0; ip < npts; ++ip) {
      TPZManVector<REAL, 3> ptInElement(gel->Dimension());
      REAL weight;
      intrule->Point(ip, ptInElement, weight);

      // Get Jacobian for integration
      TPZFMatrix<REAL> jacobian, axes, jacinv;
      REAL detjac;
      gel->Jacobian(ptInElement, jacobian, axes, detjac, jacinv);
      weight *= fabs(detjac);

      TPZManVector<REAL, 3> x(3, 0.0);
      gel->X(ptInElement, x); // Real coordinates for x

      TPZManVector<REAL, 3> force(1, 0.0);
      gexact.ForceFunc()(x, force); // Exact divergence (forcing function)

      // Compute H1 term
      TPZVec<REAL> termH1(dim,0.0);
      celH1->Solution(ptInElement, 2, termH1);
      for (int d = 0; d < dim; ++d) {
        termH1[d] = sqrt(gperm) * termH1[d];
      }

      // Compute Hdiv term and Hdiv flux divergence
      TPZVec<REAL> termHdiv(dim,0.0);
      TPZVec<REAL> divFluxMixed(1,0.0);
      TPZMultiphysicsElement *celMulti = dynamic_cast<TPZMultiphysicsElement*>(celMixed);
      if (celMulti) {
        celMulti->Solution(ptInElement, 1, termHdiv);
        celMulti->Solution(ptInElement, 5, divFluxMixed);
      }
      for (int d = 0; d < dim; ++d) {
        termHdiv[d] = (-1./sqrt(gperm)) * termHdiv[d];
      }

      // Flux contribution
      REAL diffFlux = 0.0;
      for (int d = 0; d < dim; ++d) {
        REAL diff = termH1[d] - termHdiv[d];
        diffFlux += diff * diff;
      }

      // Balance contribution
      REAL diffBalance = pow(divFluxMixed[0] - force[0], 2);

      // Add weighted contribution to element error
      fluxError += diffFlux * weight;
      balanceError += diffBalance * weight;
    }

    // For debugging purposes
    // std::cout << "Element " << gel->Index() << ": Flux error = " << fluxError
    //           << ", Balance error = " << balanceError << std::endl;

    elementError = pow((sqrt(fluxError) + (hk/(M_PI*sqrt(gperm))) * sqrt(balanceError)), 2);
    totalError += elementError;
  }

  totalError = sqrt(totalError);

  return totalError;
}

REAL ErrorEstimationMixed(TPZCompMesh *cmesh,
                        TPZMultiphysicsCompMesh *cmeshMixed, int order) {
  // Ensures references point to H1 cmesh
  cmesh->Reference()->ResetReference();
  cmesh->LoadReferences();

  int64_t ncel = cmeshMixed->NElements();
  TPZAdmChunkVector<TPZCompEl *> &elementvec_m = cmeshMixed->ElementVec();
  REAL totalError = 0.0;

  // Number of integration points for flux evaluation
  TPZIntPoints *intrule = nullptr; // Integration rule for element computations

  for (int64_t cel = 0; cel < ncel; ++cel) {

    TPZCompEl *celMixed = elementvec_m[cel];
    TPZGeoEl *gel = celMixed->Reference();
    TPZCompEl *celH1 = gel->Reference();

    if (!gel || gel->Dimension() != 2 || gel->HasSubElement())
      continue;
    if (!celMixed || !celH1)
      continue;

    REAL hk = ElementDiameter(gel); // Element size

    // Define an integration rule
    const TPZIntPoints &intruleMixed = celMixed->GetIntegrationRule();
    const TPZIntPoints &intruleH1 = celH1->GetIntegrationRule();

    if (intruleMixed.NPoints() < intruleH1.NPoints()) {
      intrule = intruleH1.Clone();
    } else {
      intrule = intruleMixed.Clone();
    }

    REAL fluxError = 0.0;
    REAL balanceError = 0.0;
    REAL conformityError = 0.0;
    REAL elementError = 0.0;
    int npts = intrule->NPoints();

    for (int ip = 0; ip < npts; ++ip) {
      TPZManVector<REAL, 3> ptInElement(gel->Dimension());
      REAL weight;
      intrule->Point(ip, ptInElement, weight);

      // Get Jacobian for integration
      TPZFMatrix<REAL> jacobian, axes, jacinv;
      REAL detjac;
      gel->Jacobian(ptInElement, jacobian, axes, detjac, jacinv);
      weight *= fabs(detjac);

      TPZManVector<REAL, 3> x(3, 0.0);
      gel->X(ptInElement, x); // Real coordinates for x

      TPZManVector<REAL, 3> force(1, 0.0);
      gexact.ForceFunc()(x, force); // Exact divergence (forcing function)

      // Compute H1 term
      TPZVec<REAL> termH1(2,0.0);
      celH1->Solution(ptInElement, 2, termH1);
      for (int d = 0; d < termH1.size(); ++d) {
        termH1[d] = sqrt(gperm) * termH1[d];
      }

      // Compute L2 term
      TPZVec<REAL> termL2(2,0.0);
      celMixed->Solution(ptInElement, 9, termL2);
      for (int d = 0; d < termL2.size(); ++d) {
        termL2[d] = sqrt(gperm) * termL2[d];
      }

      // Compute Hdiv term and Hdiv flux divergence
      TPZVec<REAL> termHdiv(2,0.0);
      TPZVec<REAL> divFluxMixed(1,0.0);
      TPZMultiphysicsElement *celMulti = dynamic_cast<TPZMultiphysicsElement*>(celMixed);
      if (celMulti) {
        celMulti->Solution(ptInElement, 1, termHdiv);
        celMulti->Solution(ptInElement, 5, divFluxMixed);
      }
      for (int d = 0; d < termHdiv.size(); ++d) {
        termHdiv[d] = (-1./sqrt(gperm)) * termHdiv[d];
      }

      // Flux contribution
      REAL diffFlux = 0.0;
      for (int d = 0; d < gel->Dimension(); ++d) {
        REAL diff = termL2[d] - termHdiv[d];
        diffFlux += diff * diff;
      }

      // Balance contribution
      REAL diffBalance = pow(divFluxMixed[0] - force[0], 2);

      // Conformity contribution
      REAL diffConformity = 0.0;
      for (int d = 0; d < termH1.size(); ++d) {
        REAL diff = termH1[d] - termL2[d];
        diffConformity += diff * diff;
      }

      // Add weighted contribution to element error
      fluxError += diffFlux * weight;
      balanceError += diffBalance * weight;
      conformityError += diffConformity * weight;
    }

    elementError = pow((sqrt(fluxError) + (hk/(M_PI*sqrt(gperm)))*sqrt(balanceError)), 2)
      + conformityError;
    totalError += elementError;

    // For debugging purposes
    // std::cout << "Element " << gel->Index() << ": Flux error = " << fluxError
    //           << ", Balance error = " << balanceError << std::endl;
  }

  totalError = sqrt(totalError);
  return totalError;
}