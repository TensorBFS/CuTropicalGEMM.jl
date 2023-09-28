#define CONCATENATE_(x, y) x##y
#define CONCATENATETHREE_(x, y, z) x##y##z

#define CONCATENATE(x, y) CONCATENATE_(x, y)
#define CONCATENATETHREE(x, y, z) CONCATENATETHREE_(x, y, z)

// The Tropical algebras
#ifdef PlusMul
#define OPERATOR_ADD(a, b) (a + b)
#define OPERATOR_MUL(a, b) (a * b)
#define PADDING 0
#define FUNCNAME _plusmul
#endif

#ifdef TropicalAndOr
#define OPERATOR_ADD(a, b) (a || b)
#define OPERATOR_MUL(a, b) (a && b)
#define PADDING false
#define FUNCNAME _andor
#endif

#ifdef TropicalMaxMul
#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUL(a, b) (a * b)
#define PADDING 0
#define FUNCNAME _maxmul
#endif

#ifdef TropicalMaxPlus
#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUL(a, b) (a + b)
#define PADDING -INFINITY
#define FUNCNAME _maxplus
#endif

#ifdef TropicalMinPlus
#define OPERATOR_ADD(a, b) min(a, b)
#define OPERATOR_MUL(a, b) (a + b)
#define PADDING INFINITY
#define FUNCNAME _minplus
#endif

// Types

#ifdef Bool
#define TYPE bool
#define TYPENAME BOOL
#endif

#ifdef FP32
#define TYPE float
#define TYPENAME FLOAT
#endif

#ifdef FP64
#define TYPE double
#define TYPENAME DOUBLE
#endif

#ifdef INT32
#define TYPE int
#define TYPENAME INT
#endif

#ifdef INT64
#define TYPE long
#define TYPENAME LONG
#endif

#ifdef S_L
#define BM 32
#define BK 32
#define BN 128
#endif

#ifdef M_M
#define BM 64
#define BK 32
#define BN 64
#endif

#ifdef L_S
#define BM 128
#define BK 32
#define BN 32
#endif

#define TT _TT
#define TN _TN
#define NT _NT
#define NN _NN