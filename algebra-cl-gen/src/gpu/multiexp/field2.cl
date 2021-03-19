#define FIELD2_LIMB_BITS FIELD_LIMB_BITS
#define FIELD2_ZERO ((FIELD2){FIELD_ZERO, FIELD_ZERO})
#define FIELD2_ONE ((FIELD2){FIELD_ONE, FIELD_ZERO})

typedef struct {
  FIELD c0;
  FIELD c1;
} FIELD2; // Represents: c0 + u * c1

bool FIELD2_eq(FIELD2 a, FIELD2 b) {
  return FIELD_eq(a.c0, b.c0) && FIELD_eq(a.c1, b.c1);
}
FIELD2 FIELD2_sub(FIELD2 a, FIELD2 b) {
  a.c0 = FIELD_sub(a.c0, b.c0);
  a.c1 = FIELD_sub(a.c1, b.c1);
  return a;
}
FIELD2 FIELD2_add(FIELD2 a, FIELD2 b) {
  a.c0 = FIELD_add(a.c0, b.c0);
  a.c1 = FIELD_add(a.c1, b.c1);
  return a;
}
FIELD2 FIELD2_double(FIELD2 a) {
  a.c0 = FIELD_double(a.c0);
  a.c1 = FIELD_double(a.c1);
  return a;
}

FIELD2 FIELD2_mul(FIELD2 a, FIELD2 b) {
  const FIELD v0 = FIELD_mul(a.c0, b.c0);
  const FIELD v1 = FIELD_mul(a.c1, b.c1);
  const FIELD o = FIELD_add(b.c0, b.c1);
  const FIELD o2 = FIELD_mul(FIELD_NONRESIDUE, v1);
  a.c1 = FIELD_add(a.c1, a.c0);
  a.c1 = FIELD_mul(a.c1, o);
  a.c1 = FIELD_sub(a.c1, v0);
  a.c1 = FIELD_sub(a.c1, v1);
  a.c0 = FIELD_add(v0, o2);
  return a;
}

FIELD2 FIELD2_sqr(FIELD2 a) {
  FIELD v0 = FIELD_sub(a.c0, a.c1);
  const FIELD o1 = FIELD_mul(FIELD_NONRESIDUE, a.c1);
  const FIELD v3 = FIELD_sub(a.c0, o1);
  const FIELD v2 = FIELD_mul(a.c0, a.c1);
  const FIELD o2 = FIELD_mul(FIELD_NONRESIDUE, v2);
  v0 = FIELD_mul(v0, v3);
  v0 = FIELD_add(v0, v2);
  a.c1 = FIELD_double(v2);
  a.c0 = FIELD_add(v0, o2);
  return a;
}
