__kernel void POINT_polycommit_round_reduce(
    __global EXPONENT* round_challenge, // 0 - normal, 1 - inv
    __global EXPONENT* round_challenge_repr, // 0 - normal, 1 - inv
    uint c_n,
    __global EXPONENT* c_l,
    __global EXPONENT* c_r,
    uint z_n,
    __global EXPONENT* z_l,
    __global EXPONENT* z_r,
    uint k_n,
    __global POINT_projective* k_l,
    __global POINT_affine* k_r
) {
    const uint gid = get_global_id(0);
    EXPONENT tmp;
    POINT_projective tmp_p;

    if (gid < c_n) {

        tmp = EXPONENT_mul(c_r[gid], round_challenge[1]);
        c_l[gid] = EXPONENT_add(c_l[gid], tmp);
    }

    if (gid < z_n) {

        tmp = EXPONENT_mul(z_r[gid], round_challenge[0]);
        z_l[gid] = EXPONENT_add(z_l[gid], tmp);
    }

    if (gid < k_n) {

        tmp_p = POINT_affine_mul(k_r[gid], round_challenge_repr[0]);
        k_l[gid] = POINT_add(k_l[gid], tmp_p);
    }
}
