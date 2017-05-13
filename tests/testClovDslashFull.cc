#include "unittest.h"
#include "testClovDslashFull.h"

// Stupid compiler thing
#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END

#include "qphix/clover_dslash_def.h"
#include "qphix/clover_dslash_body.h"

#include "qdp.h"
using namespace QDP;

#include "dslashm_w.h"
#include "reunit.h"

#include "qphix/qdp_packer.h"
#include "./clover_term.h"

#if 1
#include "qphix/clover.h"
#include "qphix/invcg.h"
#include "qphix/invbicgstab.h"
#endif

#include <omp.h>

using namespace Assertions;
using namespace std;
using namespace QPhiX;

#include "RandomGauge.h"
#include "veclen.h"
#include "tolerance.h"

template <typename FT,
          int V,
          int S,
          bool compress,
          typename QdpGauge,
          typename QdpSpinor>
void testClovDslashFull::runTest(void)
{

  typedef typename Geometry<FT, V, S, compress>::FourSpinorBlock Spinor;
  typedef typename Geometry<FT, V, S, compress>::SU3MatrixBlock Gauge;
  typedef typename Geometry<FT, V, S, compress>::CloverBlock Clover;

  bool verbose = true;
  QDPIO::cout << endl << "ENTERING CLOVER DSLASH TEST" << endl;

  // Diagnostic information:
  const multi1d<int> &lattSize = Layout::subgridLattSize();
  int Nx = lattSize[0];
  int Ny = lattSize[1];
  int Nz = lattSize[2];
  int Nt = lattSize[3];

  QDPIO::cout << "VECLEN=" << V << "  SOALEN=" << S << endl;
  QDPIO::cout << "Lattice Size: ";
  for (int mu = 0; mu < lattSize.size(); mu++) {
    QDPIO::cout << " " << lattSize[mu];
  }
  QDPIO::cout << endl;

  QDPIO::cout << "Block Sizes: By=" << By << " Bz=" << Bz << endl;
  QDPIO::cout << "N Cores" << NCores << endl;
  QDPIO::cout << "SMT Grid: Sy=" << Sy << " Sz=" << Sz << endl;
  QDPIO::cout << "Pad Factors: PadXY=" << PadXY << " PadXYZ=" << PadXYZ << endl;
  QDPIO::cout << "MinCt=" << MinCt << endl;
  QDPIO::cout << "Threads_per_core = " << N_simt << endl;

  Geometry<FT, V, S, compress> geom(Layout::subgridLattSize().slice(),
                                    By,
                                    Bz,
                                    NCores,
                                    Sy,
                                    Sz,
                                    PadXY,
                                    PadXYZ,
                                    MinCt);

  RandomGauge<FT, V, S, compress, QdpGauge, QdpSpinor> gauge(geom);

  ClovDslash<FT, V, S, compress> D32(
      &geom, gauge.t_boundary, gauge.aniso_fac_s, gauge.aniso_fac_t);

  auto psi_even = makeFourSpinorHandle(geom);
  auto psi_odd = makeFourSpinorHandle(geom);
  auto chi_even = makeFourSpinorHandle(geom);
  auto chi_odd = makeFourSpinorHandle(geom);

  QdpSpinor chi, psi;

  Spinor *psi_s[2] = {psi_even.get(), psi_odd.get()};
  Spinor *chi_s[2] = {chi_even.get(), chi_odd.get()};

  gaussian(psi);
  qdp_pack_spinor<>(psi, psi_even.get(), psi_odd.get(), geom);

#if 1
  // Test only Dslash operator.
  // For clover this will be: A^{-1}_(1-cb,1-cb) D_(1-cb, cb)  psi_cb
  QDPIO::cout << "Testing Dslash \n" << endl;

  const int cbsize_in_blocks = rb[0].numSiteTable() / S;

  for (int isign = 1; isign >= -1; isign -= 2) {
    for (int cb = 0; cb < 2; cb++) {
      int source_cb = 1 - cb;
      int target_cb = cb;

      QdpSpinor clov_chi = zero;
      QdpSpinor chi = zero;

// Apply Optimized Dslash
#if 1
      qdp_pack_spinor<>(chi, chi_even.get(), chi_odd.get(), geom);

      D32.dslash(chi_s[target_cb],
                 psi_s[source_cb],
                 gauge.u_packed[target_cb],
                 gauge.invclov_packed[target_cb],
                 isign,
                 target_cb);

      qdp_unpack_spinor<>(chi_even.get(), chi_odd.get(), clov_chi, geom);

#else
      // Work directly with the QDP-JIT pointers...
      Spinor *chi_targ = (Spinor *)(chi.getFjit()) + target_cb * cbsize_in_blocks;
      Spinor *psi_src = (Spinor *)(psi.getFjit()) + source_cb * cbsize_in_blocks;

      QDPIO::cout << "Direct Interface" << std::endl;
      Spinor *clov_chi_targ =
          (Spinor *)(clov_chi.getFjit()) + target_cb * cbsize_in_blocks;
      Spinor *psi_in_src = (Spinor *)(psi.getFjit()) + source_cb * cbsize_in_blocks;

      D32.dslash(clov_chi_targ,
                 psi_in_src,
                 u_packed[target_cb],
                 invclov_packed[target_cb],
                 isign,
                 target_cb);
#endif

      // Account for Clover term from QDP++
      // We should remove this once the actual clover implementation is
      // ready.
      // invclov_qdp.apply(clov_chi,chi,isign, target_cb);
      // clov_chi = chi;

      // Apply QDP Dslash
      QdpSpinor chi2 = zero;
      QdpSpinor clov_chi2 = zero;

      dslash(chi2, gauge.u_aniso, psi, isign, target_cb);
      gauge.invclov_qdp.apply(clov_chi2, chi2, isign, target_cb);

      // Check the difference per number in chi vector
      QdpSpinor diff = clov_chi2 - clov_chi;
      expect_near(clov_chi2,
                  clov_chi,
                  1e-6,
                  geom,
                  target_cb,
                  "Wilson clover Dslash, QPhiX vs. QDP++");

    } // cb
  } // isign

#endif

#if 1
  // Go through the test cases -- apply SSE dslash versus, QDP Dslash
  // Test ax - bDslash y
  QDPIO::cout << "Testing dslashAchiMinusBDPsi" << endl;

  for (int isign = 1; isign >= -1; isign -= 2) {
    for (int cb = 0; cb < 2; cb++) {
      int source_cb = 1 - cb;
      int target_cb = cb;

      chi = zero;
      qdp_pack_spinor<>(chi, chi_even.get(), chi_odd.get(), geom);

      double beta = (double)(0.25); // Always 0.25

      // Apply Optimized Dslash
      D32.dslashAChiMinusBDPsi(chi_s[target_cb],
                               psi_s[source_cb],
                               psi_s[target_cb],
                               gauge.u_packed[target_cb],
                               gauge.clov_packed[target_cb],
                               beta,
                               isign,
                               target_cb);

      qdp_unpack_spinor<>(chi_s[0], chi_s[1], chi, geom);

      // Apply QDP Dslash
      QdpSpinor chi2 = zero;
      dslash(chi2, gauge.u_aniso, psi, isign, target_cb);
      QdpSpinor res = zero;
      gauge.clov_qdp.apply(res, psi, isign, target_cb);
      res[rb[target_cb]] -= beta * chi2;

      // Check the difference per number in chi vector
      expect_near(res, chi, 1e-6, geom, target_cb, "A chi - b d psi, QPhiX vs. QDP++");
    }
  }

#endif

#if 1
  // Test only Dslash operator.
  // For clover this will be: A^{-1}_(1-cb,1-cb) D_(1-cb, cb)  psi_cb
  QDPIO::cout << "Testing Dslash With antiperiodic BCs \n" << endl;

  RandomGauge<FT, V, S, compress, QdpGauge, QdpSpinor> gauge_antip(geom, -1.0);

  // Create Antiperiodic Dslash
  ClovDslash<FT, V, S, compress> D32_ap(&geom,
                                        gauge_antip.t_boundary,
                                        gauge_antip.aniso_fac_s,
                                        gauge_antip.aniso_fac_t);

  for (int isign = 1; isign >= -1; isign -= 2) {
    for (int cb = 0; cb < 2; cb++) {
      int source_cb = 1 - cb;
      int target_cb = cb;

      QdpSpinor clov_chi = zero;
      qdp_pack_spinor<>(chi, chi_even.get(), chi_odd.get(), D32_ap.getGeometry());

      // Apply Optimized Dslash
      D32_ap.dslash(chi_s[target_cb],
                    psi_s[source_cb],
                    gauge_antip.u_packed[target_cb],
                    gauge_antip.invclov_packed[target_cb],
                    isign,
                    target_cb);

      qdp_unpack_spinor<>(chi_even.get(), chi_odd.get(), clov_chi, geom);

      // Account for Clover term from QDP++
      // We should remove this once the actual clover implementation is
      // ready.
      // invclov_qdp.apply(clov_chi,chi,isign, target_cb);
      // clov_chi = chi;

      // Apply QDP Dslash
      QdpSpinor chi2 = zero;
      QdpSpinor clov_chi2 = zero;
      dslash(chi2, gauge_antip.u_aniso, psi, isign, target_cb);
      gauge_antip.invclov_qdp.apply(clov_chi2, chi2, isign, target_cb);

      expect_near(clov_chi2,
                  clov_chi,
                  1e-6,
                  geom,
                  target_cb,
                  "Dslash with antiperiodic T, QPhiX vs. QDP++");
    } // cb
  } // isign

#endif

#if 1
  // Go through the test cases -- apply SSE dslash versus, QDP Dslash
  // Test ax - bDslash y
  QDPIO::cout << "Testing dslashAchiMinusBDPsi" << endl;

  for (int isign = 1; isign >= -1; isign -= 2) {
    for (int cb = 0; cb < 2; cb++) {
      int source_cb = 1 - cb;
      int target_cb = cb;

      chi = zero;
      qdp_pack_spinor<>(chi, chi_even.get(), chi_odd.get(), D32_ap.getGeometry());

      double beta = (double)(0.25); // Always 0.25

      // Apply Optimized Dslash
      D32_ap.dslashAChiMinusBDPsi(chi_s[target_cb],
                                  psi_s[source_cb],
                                  psi_s[target_cb],
                                  gauge_antip.u_packed[target_cb],
                                  gauge_antip.clov_packed[target_cb],
                                  beta,
                                  isign,
                                  target_cb);

      qdp_unpack_spinor<>(chi_s[0], chi_s[1], chi, D32_ap.getGeometry());

      // Apply QDP Dslash
      QdpSpinor chi2 = zero;
      QdpSpinor clov_chi2 = zero;
      dslash(chi2, gauge_antip.u_aniso, psi, isign, target_cb);
      QdpSpinor res = zero;
      gauge_antip.clov_qdp.apply(res, psi, isign, target_cb);
      res[rb[target_cb]] -= beta * chi2;

      expect_near(res,
                  chi,
                  1e-6,
                  geom,
                  target_cb,
                  "Antiperiodic A chi - b d psi, QPhiX vs. QDP++");
    }
  }

#endif

  // Disabling testing the even odd operator until recoded with new
  // vectorization
  //

  QdpSpinor ltmp = zero;
  Real betaFactor = Real(0.25);
#if 1
  for (int target_cb = 0; target_cb < 2; ++target_cb) {
    int source_cb = 1 - target_cb;
    // FIXME: cb is needed to make the even odd operator... This has effects
    // down the line

    QDPIO::cout << "Testing Even Odd Operator" << endl;
    EvenOddCloverOperator<FT, V, S, compress> M(
        gauge_antip.u_packed,
        gauge_antip.clov_packed[target_cb],
        gauge_antip.invclov_packed[source_cb],
        &geom,
        gauge_antip.t_boundary,
        gauge_antip.aniso_fac_s,
        gauge_antip.aniso_fac_t);
    // Apply optimized
    for (int isign = 1; isign >= -1; isign -= 2) {
      chi = zero;
      qdp_pack_spinor<>(chi, chi_even.get(), chi_odd.get(), geom);

      M(chi_s[target_cb], psi_s[target_cb], isign, target_cb);

      qdp_unpack_spinor<>(chi_s[0], chi_s[1], chi, geom);

      // Apply QDP Dslash
      QdpSpinor chi2 = zero;
      QdpSpinor clov_chi2 = zero;

      dslash(chi2, gauge_antip.u_aniso, psi, isign, source_cb);
      gauge_antip.invclov_qdp.apply(clov_chi2, chi2, isign, source_cb);
      dslash(ltmp, gauge_antip.u_aniso, clov_chi2, isign, target_cb);

      gauge_antip.clov_qdp.apply(chi2, psi, isign, target_cb);

      chi2[rb[target_cb]] -= betaFactor * ltmp;

      expect_near(
          chi2, chi, 1e-6, geom, target_cb, "Even-odd operator, QPhiX vs. QDP++");
    } // isign loop

  } // cb loop

#endif

#if 1
  {
    for (int cb = 0; cb < 2; ++cb) {
      int other_cb = 1 - cb;
      EvenOddCloverOperator<FT, V, S, compress> M(gauge_antip.u_packed,
                                                  gauge_antip.clov_packed[cb],
                                                  gauge_antip.invclov_packed[other_cb],
                                                  &geom,
                                                  gauge_antip.t_boundary,
                                                  gauge_antip.aniso_fac_s,
                                                  gauge_antip.aniso_fac_t);

      chi = zero;
      qdp_pack_cb_spinor<>(chi, chi_s[cb], geom, cb);

      double rsd_target = rsdTarget<FT>::value;
      int max_iters = 500;
      int niters;
      double rsd_final;
      unsigned long site_flops;
      unsigned long mv_apps;

      InvCG<FT, V, S, compress> solver(M, max_iters);

      double start = omp_get_wtime();
      solver(chi_s[cb],
             psi_s[cb],
             rsd_target,
             niters,
             rsd_final,
             site_flops,
             mv_apps,
             1,
             verbose,
             cb);
      double end = omp_get_wtime();

      qdp_unpack_cb_spinor<>(chi_s[cb], chi, geom, cb);

      QdpSpinor chi2, clov_chi2;

      // Multiply back
      // chi2 = M chi
      dslash(chi2, gauge_antip.u_aniso, chi, 1, other_cb);
      gauge_antip.invclov_qdp.apply(clov_chi2, chi2, 1, other_cb);
      dslash(ltmp, gauge_antip.u_aniso, clov_chi2, 1, cb);

      gauge_antip.clov_qdp.apply(chi2, chi, 1, cb);
      chi2[rb[cb]] -= betaFactor * ltmp;

      // chi3 = M^\dagger chi2
      QdpSpinor chi3 = zero;
      dslash(chi3, gauge_antip.u_aniso, chi2, -1, other_cb);
      gauge_antip.invclov_qdp.apply(clov_chi2, chi3, -1, other_cb);
      dslash(ltmp, gauge_antip.u_aniso, clov_chi2, -1, cb);

      gauge_antip.clov_qdp.apply(chi3, chi2, -1, cb);
      chi3[rb[cb]] -= betaFactor * ltmp;

      //  dslash(chi3,u,chi2, (-1), 1);
      // dslash(ltmp,u,chi3, (-1), 0);
      // chi3[rb[0]] = massFactor*chi2 - betaFactor*ltmp;

      QdpSpinor diff = chi3 - psi;
      QDPIO::cout << "cb=" << cb << " True norm is: "
                  << sqrt(norm2(diff, rb[cb]) / norm2(psi, rb[cb])) << endl;

      expect_near(chi3, psi, 1e-6, geom, cb, "CG");

      int Nxh = Nx / 2;
      unsigned long num_cb_sites = Nxh * Ny * Nz * Nt;

      unsigned long total_flops =
          (site_flops + (1320 + 504 + 1320 + 504 + 48) * mv_apps) * num_cb_sites;

      masterPrintf("GFLOPS=%e\n", 1.0e-9 * (double)(total_flops) / (end - start));
    }
  }
#endif

#if 1
  {
    for (int cb = 0; cb < 2; ++cb) {
      int other_cb = 1 - cb;
      EvenOddCloverOperator<FT, V, S, compress> M(gauge_antip.u_packed,
                                                  gauge_antip.clov_packed[cb],
                                                  gauge_antip.invclov_packed[other_cb],
                                                  &geom,
                                                  gauge_antip.t_boundary,
                                                  gauge_antip.aniso_fac_s,
                                                  gauge_antip.aniso_fac_t);

      chi = zero;
      qdp_pack_spinor<>(chi, chi_even.get(), chi_odd.get(), geom);

      double rsd_target = rsdTarget<FT>::value;
      int max_iters = 500;
      int niters;
      double rsd_final;
      unsigned long site_flops;
      unsigned long mv_apps;

      InvBiCGStab<FT, V, S, compress> solver(M, max_iters);
      const int isign = 1;
      double start = omp_get_wtime();
      solver(chi_s[cb],
             psi_s[cb],
             rsd_target,
             niters,
             rsd_final,
             site_flops,
             mv_apps,
             isign,
             verbose,
             cb);
      double end = omp_get_wtime();

      qdp_unpack_cb_spinor<>(chi_s[cb], chi, geom, cb);

      // Multiply back
      // chi2 = M chi
      QdpSpinor chi2, clov_chi, clov_chi2;
      dslash(chi2, gauge_antip.u_aniso, chi, 1, other_cb);
      gauge_antip.invclov_qdp.apply(clov_chi2, chi2, 1, other_cb);
      dslash(ltmp, gauge_antip.u_aniso, clov_chi2, 1, cb);

      gauge_antip.clov_qdp.apply(chi2, chi, 1, cb);
      chi2[rb[cb]] -= betaFactor * ltmp;

      QdpSpinor diff = chi2 - psi;
      QDPIO::cout << " cb = " << cb << " True norm is: "
                  << sqrt(norm2(diff, rb[cb]) / norm2(psi, rb[cb])) << endl;
      expect_near(chi2, psi, 1e-6, geom, cb, "BiCGStab");

      int Nxh = Nx / 2;
      unsigned long num_cb_sites = Nxh * Ny * Nz * Nt;

      unsigned long total_flops =
          (site_flops + (1320 + 504 + 1320 + 504 + 48) * mv_apps) * num_cb_sites;
      masterPrintf("GFLOPS=%e\n", 1.0e-9 * (double)(total_flops) / (end - start));
    } // cb loop
  } // scope
#endif
}

void testClovDslashFull::run(void)
{
  typedef LatticeColorMatrixF UF;
  typedef LatticeDiracFermionF PhiF;

  typedef LatticeColorMatrixD UD;
  typedef LatticeDiracFermionD PhiD;

#if defined(QPHIX_SCALAR_SOURCE)
  if (precision == FLOAT_PREC) {
    QDPIO::cout << "SINGLE PRECISION TESTING " << endl;
    if (compress12) {
      runTest<float, 1, 1, true, UF, PhiF>();
    } else {
      runTest<float, 1, 1, false, UF, PhiF>();
    }
  }
  if (precision == DOUBLE_PREC) {
    QDPIO::cout << "DOUBLE PRECISION TESTING " << endl;
    if (compress12) {
      runTest<double, 1, 1, true, UF, PhiF>();
    } else {
      runTest<double, 1, 1, false, UF, PhiF>();
    }
  }
#else

  if (precision == FLOAT_PREC) {
#if defined(QPHIX_AVX_SOURCE) || defined(QPHIX_AVX2_SOURCE) ||                      \
    defined(QPHIX_MIC_SOURCE) || defined(QPHIX_AVX512_SOURCE) ||                    \
    defined(QPHIX_SSE_SOURCE)
    QDPIO::cout << "SINGLE PRECISION TESTING " << endl;
    if (compress12) {
      runTest<float, VECLEN_SP, 4, true, UF, PhiF>();
    } else {
      runTest<float, VECLEN_SP, 4, false, UF, PhiF>();
    }

#if !defined(QPHIX_SSE_SOURCE)
    if (compress12) {
      runTest<float, VECLEN_SP, 8, true, UF, PhiF>();
    } else {
      runTest<float, VECLEN_SP, 8, false, UF, PhiF>();
    }
#endif

#if defined(QPHIX_MIC_SOURCE) || defined(QPHIX_AVX512_SOURCE)
    if (compress12) {
      runTest<float, VECLEN_SP, 16, true, UF, PhiF>();
    } else {
      runTest<float, VECLEN_SP, 16, false, UF, PhiF>();
    }
#endif
#endif // QPHIX_AVX_SOURCE|| QPHIX_AVX2_SOURCE|| QPHIX_MIC_SOURCE |
    // QPHIX_AVX512_SOURCE
  }

  if (precision == HALF_PREC) {
#if defined(QPHIX_MIC_SOURCE) || defined(QPHIX_AVX512_SOURCE) ||                    \
    defined(QPHIX_AVX2_SOURCE)
    QDPIO::cout << "SINGLE PRECISION TESTING " << endl;
    if (compress12) {
      runTest<half, VECLEN_HP, 4, true, UF, PhiF>();
    } else {
      runTest<half, VECLEN_HP, 4, false, UF, PhiF>();
    }

    if (compress12) {
      runTest<half, VECLEN_HP, 8, true, UF, PhiF>();
    } else {
      runTest<half, VECLEN_HP, 8, false, UF, PhiF>();
    }

#if defined(QPHIX_MIC_SOURCE) || defined(QPHIX_AVX512_SOURCE)
    if (compress12) {
      runTest<half, VECLEN_HP, 16, true, UF, PhiF>();
    } else {
      runTest<half, VECLEN_HP, 16, false, UF, PhiF>();
    }
#endif
#else
    QDPIO::cout << "Half precision tests are not available in this build. "
                   "Currently only in MIC builds"
                << endl;
#endif
  }

  if (precision == DOUBLE_PREC) {
    QDPIO::cout << "DOUBLE PRECISION TESTING" << endl;

#if defined(QPHIX_AVX_SOURCE) || defined(QPHIX_AVX2_SOURCE)
    // Only AVX can do DP 2
    if (compress12) {
      runTest<double, VECLEN_DP, 2, true, UD, PhiD>();
    } else {
      runTest<double, VECLEN_DP, 2, false, UD, PhiD>();
    }
#endif

    if (compress12) {
      runTest<double, VECLEN_DP, 4, true, UD, PhiD>();
    } else {
      runTest<double, VECLEN_DP, 4, false, UD, PhiD>();
    }

#if defined(QPHIX_MIC_SOURCE) || defined(QPHIX_AVX512_SOURCE)
    // Only MIC can do DP 8
    if (compress12) {
      runTest<double, VECLEN_DP, 8, true, UD, PhiD>();
    } else {
      runTest<double, VECLEN_DP, 8, false, UD, PhiD>();
    }
#endif
  }

#endif // SCALAR SOURCE
}
