#if !defined(FPTYPE)
#error FTYPE not defined
#endif

#if !defined(VEC)
#error VLEN not defined
#endif

template <>
inline void dslash_plus_vec<FPTYPE, VEC, SOA, COMPRESS12>(
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *xyBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zfBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tfBase,
    Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *oBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::SU3MatrixBlock *gBase,
    const int xbOffs[VEC],
    const int xfOffs[VEC],
    const int ybOffs[VEC],
    const int yfOffs[VEC],
    const int offs[VEC],
    const int gOffs[VEC],
    const int siprefdist1,
    const int siprefdist2,
    const int siprefdist3,
    const int siprefdist4,
    const int gprefdist,
    const int pfyOffs[VEC],
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase2,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase3,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase4,
    const unsigned int accumulate[8],
    const FPTYPE coeff_s,
    const FPTYPE coeff_t_f,
    const FPTYPE coeff_t_b) {
    // clang-format off
{{ include_generated_kernel(ISA, kernel, "plus_body", FPTYPE, VEC, SOA, COMPRESS12) }}
    // clang-format on
}

template <>
inline void dslash_minus_vec<FPTYPE, VEC, SOA, COMPRESS12>(
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *xyBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zfBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tfBase,
    Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *oBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::SU3MatrixBlock *gBase,
    const int xbOffs[VEC],
    const int xfOffs[VEC],
    const int ybOffs[VEC],
    const int yfOffs[VEC],
    const int offs[VEC],
    const int gOffs[VEC],
    const int siprefdist1,
    const int siprefdist2,
    const int siprefdist3,
    const int siprefdist4,
    const int gprefdist,
    const int pfyOffs[VEC],
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase2,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase3,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase4,
    const unsigned int accumulate[8],
    const FPTYPE coeff_s,
    const FPTYPE coeff_t_f,
    const FPTYPE coeff_t_b) {
    // clang-format off
{{ include_generated_kernel(ISA, kernel, "minus_body", FPTYPE, VEC, SOA, COMPRESS12) }}
    // clang-format on
}

template <>
inline void dslash_achimbdpsi_plus_vec<FPTYPE, VEC, SOA, COMPRESS12>(
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *xyBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zfBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tfBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *chiBase,
    Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *oBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::SU3MatrixBlock *gBase,
    const int xbOffs[VEC],
    const int xfOffs[VEC],
    const int ybOffs[VEC],
    const int yfOffs[VEC],
    const int offs[VEC],
    const int gOffs[VEC],
    const int siprefdist1,
    const int siprefdist2,
    const int siprefdist3,
    const int siprefdist4,
    const int chiprefdist,
    const int gprefdist,
    const int pfyOffs[VEC],
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase2,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase3,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase4,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBaseChi,
    const FPTYPE alpha,
    const FPTYPE coeff_s,
    const FPTYPE coeff_t_f,
    const FPTYPE coeff_t_b,
    const unsigned int accumulate[8]) {
    // clang-format off
{{ include_generated_kernel(ISA, kernel, "achimbdpsi_plus_body", FPTYPE, VEC, SOA, COMPRESS12) }}
    // clang-format on
}

template <>
inline void dslash_achimbdpsi_minus_vec<FPTYPE, VEC, SOA, COMPRESS12>(
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *xyBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *zfBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tbBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *tfBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *chiBase,
    Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *oBase,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::SU3MatrixBlock *gBase,
    const int xbOffs[VEC],
    const int xfOffs[VEC],
    const int ybOffs[VEC],
    const int yfOffs[VEC],
    const int offs[VEC],
    const int gOffs[VEC],
    const int siprefdist1,
    const int siprefdist2,
    const int siprefdist3,
    const int siprefdist4,
    const int chiprefdist,
    const int gprefdist,
    const int pfyOffs[VEC],
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase2,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase3,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBase4,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *pfBaseChi,
    const FPTYPE alpha,
    const FPTYPE coeff_s,
    const FPTYPE coeff_t_f,
    const FPTYPE coeff_t_b,
    const unsigned int accumulate[8]) {
    // clang-format off
{{ include_generated_kernel(ISA, kernel, "achimbdpsi_minus_body", FPTYPE, VEC, SOA, COMPRESS12) }}
    // clang-format on
}

#ifdef QPHIX_DO_COMMS

template <>
inline void face_proj_dir_plus<FPTYPE, VEC, SOA, COMPRESS12>(
    const typename Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock
        *xyBase,
    const int offs[VEC],
    const int si_prefdist,
    FPTYPE *outbuf,
    const int hsprefdist,
    unsigned int mask,
    int dir) {
    if (dir == 0) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_X_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 1) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_X_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 2) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_Y_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 3) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_Y_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 4) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_Z_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 5) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_Z_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 6) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_T_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 7) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_T_plus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else {
        printf("Invalid dir for pack boundary\n");
        exit(1);
    }
}

template <>
inline void face_proj_dir_minus<FPTYPE, VEC, SOA, COMPRESS12>(
    const typename Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock
        *xyBase,
    const int offs[VEC],
    const int si_prefdist,
    FPTYPE *outbuf,
    const int hsprefdist,
    unsigned int mask,
    int dir) {
    if (dir == 0) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_X_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 1) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_X_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 2) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_Y_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 3) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_Y_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 4) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_Z_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 5) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_Z_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 6) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_back_T_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else if (dir == 7) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_pack_to_forw_T_minus", FPTYPE, VEC, SOA, "") }}
        // clang-format on
    } else {
        printf("Invalid dir for pack boundary\n");
        exit(1);
    }
}

template <>
inline void face_finish_dir_plus<FPTYPE, VEC, SOA, COMPRESS12>(
    const FPTYPE *inbuf,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::SU3MatrixBlock *gBase,
    Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *oBase,
    const int gOffs[VEC],
    const int offs[VEC],
    const int hsprefdist,
    const int gprefdist,
    const int soprefdist,
    const FPTYPE beta,
    unsigned int mask,
    int dir) {
    if (dir == 0) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_X_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 1) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_X_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 2) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_Y_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 3) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_Y_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 4) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_Z_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 5) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_Z_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 6) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_T_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 7) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_T_plus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else {
        printf("Invalid dir for unpack boundary\n");
        exit(1);
    }
}

template <>
inline void face_finish_dir_minus<FPTYPE, VEC, SOA, COMPRESS12>(
    const FPTYPE *inbuf,
    const Geometry<FPTYPE, VEC, SOA, COMPRESS12>::SU3MatrixBlock *gBase,
    Geometry<FPTYPE, VEC, SOA, COMPRESS12>::FourSpinorBlock *oBase,
    const int gOffs[VEC],
    const int offs[VEC],
    const int hsprefdist,
    const int gprefdist,
    const int soprefdist,
    const FPTYPE beta,
    unsigned int mask,
    int dir) {
    if (dir == 0) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_X_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 1) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_X_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 2) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_Y_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 3) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_Y_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 4) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_Z_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 5) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_Z_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 6) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_back_T_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else if (dir == 7) {
        // clang-format off
{{ include_generated_kernel(ISA, kernel, "face_unpack_from_forw_T_minus", FPTYPE, VEC, SOA, COMPRESS12) }}
        // clang-format on
    } else {
        printf("Invalid dir for unpack boundary\n");
        exit(1);
    }
}

#endif // QPHIX_DO_COMMS (outer)
