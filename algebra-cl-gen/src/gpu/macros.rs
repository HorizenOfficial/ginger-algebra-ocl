macro_rules! multiexp_src {
    (
        $Limb64:expr, $Curve:ident, false,
        $Point1:expr, $Base1:expr, $Scalar1:expr, 
        $Point2:expr, $Base2:expr, $Scalar2:expr
    ) => (
        vec![
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb32>("Fr")
            },
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb32>("Fq")
            },
            ec($Base1, $Point1, $Scalar1),
            multiexp($Point1, $Scalar1),
            ec($Base2, $Point2, $Scalar2),
            multiexp($Point2, $Scalar2),
        ]
        .join("\n\n")
    );
    (
        $Limb64:expr, $Curve:ident, true,
        $Point1:expr, $Base1:expr, $Scalar1:expr, 
        $Point2:expr, $Base2:expr, $Scalar2:expr
    ) => (
        vec![
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb32>("Fr")
            },
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb32>("Fq")
            },
            if $Limb64 {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb32>("Fq2", "Fq")
            },            
            field2("Fq2", "Fq"),
            ec($Base1, $Point1, $Scalar1),
            multiexp($Point1, $Scalar1),
            ec($Base2, $Point2, $Scalar2),
            multiexp($Point2, $Scalar2),
        ]
        .join("\n\n")
    );
    (
        $Limb64:expr, $Curve:ident, true,
        $Point1:expr, $Base1:expr, $Scalar1:expr, 
        $Point2:expr, $Base2:expr, $Scalar2:expr,
        $Point3:expr, $Base3:expr, $Scalar3:expr
    ) => (
        vec![
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb32>("Fr")
            },
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb32>("Fq")
            },
            if $Limb64 {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb32>("Fq2", "Fq")
            },            
            field2("Fq2", "Fq"),
            ec($Base1, $Point1, $Scalar1),
            multiexp($Point1, $Scalar1),
            ec($Base2, $Point2, $Scalar2),
            multiexp($Point2, $Scalar2),
            ec($Base3, $Point3, $Scalar3),
            multiexp($Point3, $Scalar3),
        ]
        .join("\n\n")
    );
}

macro_rules! polycommit_round_reduce_src {
    (
        $Limb64:expr, $Curve:ident, false,
        $Point1:expr, $Base1:expr, $Scalar1:expr, 
        $Point2:expr, $Base2:expr, $Scalar2:expr
    ) => (
        vec![
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb32>("Fr")
            },
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb32>("Fq")
            },
            ec($Base1, $Point1, $Scalar1),
            polycommit_round_reduce($Point1, $Scalar1),
            ec($Base2, $Point2, $Scalar2),
            polycommit_round_reduce($Point2, $Scalar2),
        ]
        .join("\n\n")
    );
    (
        $Limb64:expr, $Curve:ident, true,
        $Point1:expr, $Base1:expr, $Scalar1:expr, 
        $Point2:expr, $Base2:expr, $Scalar2:expr
    ) => (
        vec![
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb32>("Fr")
            },
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb32>("Fq")
            },
            if $Limb64 {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb32>("Fq2", "Fq")
            },            
            field2("Fq2", "Fq"),
            ec($Base1, $Point1, $Scalar1),
            polycommit_round_reduce($Point1, $Scalar1),
            ec($Base2, $Point2, $Scalar2),
            polycommit_round_reduce($Point2, $Scalar2),
        ]
        .join("\n\n")
    );
    (
        $Limb64:expr, $Curve:ident, true,
        $Point1:expr, $Base1:expr, $Scalar1:expr, 
        $Point2:expr, $Base2:expr, $Scalar2:expr,
        $Point3:expr, $Base3:expr, $Scalar3:expr
    ) => (
        vec![
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fr, Limb32>("Fr")
            },
            if $Limb64 {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::$Curve::Fq, Limb32>("Fq")
            },
            if $Limb64 {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::$Curve::Fq2Parameters, Limb32>("Fq2", "Fq")
            },            
            field2("Fq2", "Fq"),
            ec($Base1, $Point1, $Scalar1),
            polycommit_round_reduce($Point1, $Scalar1),
            ec($Base2, $Point2, $Scalar2),
            polycommit_round_reduce($Point2, $Scalar2),
            ec($Base3, $Point3, $Scalar3),
            polycommit_round_reduce($Point3, $Scalar3),
        ]
        .join("\n\n")
    );
}
