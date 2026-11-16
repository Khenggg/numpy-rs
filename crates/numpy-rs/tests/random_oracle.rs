mod common;

use common::{
    assert_aligned, assert_f32_close, assert_f64_close, py_assign_f64, py_assign_dyn, run_oracle,
    sample_mean_var,
};
use numpy_random::{
    default_rng_from_seed, AnyBitGenerator, BitGenerator, CheckValid, Generator, MvHypergeomMethod,
    MvNormalMethod, RandomState, SeedSequence, MT19937, PCG64, PCG64DXSM, Philox, SFC64,
};
use numpy_types::{DynArray, NdArray};

fn check_f64_arr(
    actual: &NdArray<f64>,
    oracle: &common::OracleOutput,
    tag: &str,
    tol: f64,
) {
    assert_aligned(actual, tag);
    let exp = oracle.arr(tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{tag}: shape mismatch");
    assert_f64_close(&actual.to_vec(), &exp.to_f64_vec(), tol, tol, false, tag);
}

fn check_f32_arr(
    actual: &NdArray<f32>,
    oracle: &common::OracleOutput,
    tag: &str,
    tol: f64,
) {
    assert_aligned(actual, tag);
    let exp = oracle.arr(tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{tag}: shape mismatch");
    assert_f32_close(&actual.to_vec(), &exp.to_f32_vec(), tol, tol, false, tag);
}

fn check_i64_arr(actual: &NdArray<i64>, oracle: &common::OracleOutput, tag: &str) {
    assert_aligned(actual, tag);
    let exp = oracle.arr(tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{tag}: shape mismatch");
    assert_eq!(actual.to_vec(), exp.to_i64_vec(), "{tag}: value mismatch");
}

#[test]
fn test_seed_sequence_and_bit_generators_against_oracle() {
    let oracle = run_oracle(
        r#"
ss1 = np.random.SeedSequence(0xdeadbeaf)
emit_arr("ss1_u32", ss1.generate_state(8, dtype=np.uint32))
emit_arr("ss1_u64", ss1.generate_state(4, dtype=np.uint64))

ss_arr = np.random.SeedSequence([0x12345678, 0x9abcdef0, 42])
emit_arr("ssa_u32", ss_arr.generate_state(6, dtype=np.uint32))
kids = ss_arr.spawn(3)
for i, k in enumerate(kids):
    emit_arr(f"ssa_kid{i}", k.generate_state(4, dtype=np.uint64))
gkids = kids[1].spawn(2)
for j, gk in enumerate(gkids):
    emit_arr(f"ssa_gk{j}", gk.generate_state(4, dtype=np.uint32))

ss_empty = np.random.SeedSequence([])
emit_arr("ss_empty_u32", ss_empty.generate_state(4, dtype=np.uint32))

ss_u128 = np.random.SeedSequence(0x0123456789abcdef_fedcba9876543210)
emit_arr("ss_u128_u64", ss_u128.generate_state(4, dtype=np.uint64))

seed = 424242
for name, cls in [
    ("pcg64", np.random.PCG64),
    ("dxsm", np.random.PCG64DXSM),
    ("mt", np.random.MT19937),
    ("philox", np.random.Philox),
    ("sfc64", np.random.SFC64),
]:
    bg = cls(np.random.SeedSequence(seed))
    emit_arr(f"{name}_raw", np.array([bg.random_raw() for _ in range(10)], dtype=np.uint64))
    g = np.random.Generator(cls(np.random.SeedSequence(seed)))
    emit_arr(f"{name}_f64", g.random(6))
    emit_arr(f"{name}_f32", g.random(6, dtype=np.float32))

# jumped / advance checks
pcg = np.random.PCG64(np.random.SeedSequence(seed))
pcg_j = pcg.jumped(2)
emit_arr("pcg_j2", np.array([pcg_j.random_raw() for _ in range(4)], dtype=np.uint64))
pcg_adv = np.random.PCG64(np.random.SeedSequence(seed)).advance(5)
emit_arr("pcg_adv5", np.array([pcg_adv.random_raw() for _ in range(4)], dtype=np.uint64))

dx = np.random.PCG64DXSM(np.random.SeedSequence(seed))
dx_j = dx.jumped(1)
emit_arr("dx_j1", np.array([dx_j.random_raw() for _ in range(4)], dtype=np.uint64))

ph = np.random.Philox(np.random.SeedSequence(seed))
ph_j = ph.jumped(1)
emit_arr("ph_j1", np.array([ph_j.random_raw() for _ in range(4)], dtype=np.uint64))

mt_leg = np.random.MT19937()
mt_leg._legacy_seeding(12345)
emit_arr("mt_leg_raw", np.array([mt_leg.random_raw() for _ in range(8)], dtype=np.uint64))
"#,
    );

    // SeedSequence checks
    let ss1 = SeedSequence::from_seed(0xdeadbeafu64).unwrap();
    assert_eq!(ss1.generate_state_u32(8), oracle.arr("ss1_u32").to_u32_vec());
    assert_eq!(ss1.generate_state_u64(4), oracle.arr("ss1_u64").to_u64_vec());

    let mut ss_arr = SeedSequence::from_seed([0x1234_5678u32, 0x9abc_def0, 42]).unwrap();
    assert_eq!(ss_arr.generate_state_u32(6), oracle.arr("ssa_u32").to_u32_vec());
    let mut kids = ss_arr.spawn(3).unwrap();
    for (i, k) in kids.iter().enumerate() {
        assert_eq!(k.generate_state_u64(4), oracle.arr(&format!("ssa_kid{i}")).to_u64_vec());
    }
    let gkids = kids[1].spawn(2).unwrap();
    for (j, gk) in gkids.iter().enumerate() {
        assert_eq!(gk.generate_state_u32(4), oracle.arr(&format!("ssa_gk{j}")).to_u32_vec());
    }

    let ss_empty = SeedSequence::from_seed(Vec::<u32>::new()).unwrap();
    assert_eq!(ss_empty.generate_state_u32(4), oracle.arr("ss_empty_u32").to_u32_vec());

    let ss_u128 = SeedSequence::from_seed(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128).unwrap();
    assert_eq!(ss_u128.generate_state_u64(4), oracle.arr("ss_u128_u64").to_u64_vec());

    assert!(SeedSequence::from_seed(-1i64).is_err());
    assert!(SeedSequence::with_options(1u32, Vec::new(), 3, 0).is_err());

    let seed = 424242u64;
    // PCG64
    let mut pcg = PCG64::from_seed(seed).unwrap();
    assert_eq!(pcg.random_raw(10), oracle.arr("pcg64_raw").to_u64_vec());
    let mut g_pcg = Generator::new(PCG64::from_seed(seed).unwrap());
    check_f64_arr(&g_pcg.random(&[6]).unwrap(), &oracle, "pcg64_f64", 1e-15);
    check_f32_arr(&g_pcg.random_f32(&[6]).unwrap(), &oracle, "pcg64_f32", 1e-7);

    // PCG64DXSM
    let mut dx = PCG64DXSM::from_seed(seed).unwrap();
    assert_eq!(dx.random_raw(10), oracle.arr("dxsm_raw").to_u64_vec());
    let mut g_dx = Generator::new(PCG64DXSM::from_seed(seed).unwrap());
    check_f64_arr(&g_dx.random(&[6]).unwrap(), &oracle, "dxsm_f64", 1e-15);
    check_f32_arr(&g_dx.random_f32(&[6]).unwrap(), &oracle, "dxsm_f32", 1e-7);

    // MT19937
    let mut mt = MT19937::from_seed(seed).unwrap();
    assert_eq!(mt.random_raw(10), oracle.arr("mt_raw").to_u64_vec());
    let mut g_mt = Generator::new(MT19937::from_seed(seed).unwrap());
    check_f64_arr(&g_mt.random(&[6]).unwrap(), &oracle, "mt_f64", 1e-15);
    check_f32_arr(&g_mt.random_f32(&[6]).unwrap(), &oracle, "mt_f32", 1e-7);

    // Philox
    let mut ph = Philox::from_seed(seed).unwrap();
    assert_eq!(ph.random_raw(10), oracle.arr("philox_raw").to_u64_vec());
    let mut g_ph = Generator::new(Philox::from_seed(seed).unwrap());
    check_f64_arr(&g_ph.random(&[6]).unwrap(), &oracle, "philox_f64", 1e-15);
    check_f32_arr(&g_ph.random_f32(&[6]).unwrap(), &oracle, "philox_f32", 1e-7);

    // SFC64
    let mut sfc = SFC64::from_seed(seed).unwrap();
    assert_eq!(sfc.random_raw(10), oracle.arr("sfc64_raw").to_u64_vec());
    let mut g_sfc = Generator::new(SFC64::from_seed(seed).unwrap());
    check_f64_arr(&g_sfc.random(&[6]).unwrap(), &oracle, "sfc64_f64", 1e-15);
    check_f32_arr(&g_sfc.random_f32(&[6]).unwrap(), &oracle, "sfc64_f32", 1e-7);

    // Jump & Advance checks
    let mut pcg_j = PCG64::from_seed(seed).unwrap().jumped(2);
    assert_eq!(pcg_j.random_raw(4), oracle.arr("pcg_j2").to_u64_vec());
    let mut pcg_a = PCG64::from_seed(seed).unwrap();
    pcg_a.advance(5);
    assert_eq!(pcg_a.random_raw(4), oracle.arr("pcg_adv5").to_u64_vec());

    let mut dx_j = PCG64DXSM::from_seed(seed).unwrap().jumped(1);
    assert_eq!(dx_j.random_raw(4), oracle.arr("dx_j1").to_u64_vec());

    let mut ph_j = Philox::from_seed(seed).unwrap().jumped(1);
    assert_eq!(ph_j.random_raw(4), oracle.arr("ph_j1").to_u64_vec());

    let mut mt_leg = MT19937::from_legacy_seed(12345);
    assert_eq!(mt_leg.random_raw(8), oracle.arr("mt_leg_raw").to_u64_vec());

    // AnyBitGenerator checks
    let any_s = AnyBitGenerator::SFC64(SFC64::from_seed(seed).unwrap());
    let mut g_any = Generator::new(any_s);
    check_f64_arr(&g_any.random(&[6]).unwrap(), &oracle, "sfc64_f64", 1e-15);
}

#[test]
fn test_generator_all_distributions_against_oracle() {
    let oracle = run_oracle(
        r#"
seed = 20261116
def g():
    return np.random.default_rng(seed)

emit_arr("bytes", np.frombuffer(g().bytes(13), dtype=np.uint8))
emit_arr("int_i64", g().integers(-15, 25, size=(3, 4)))
emit_arr("int_incl", g().integers(0, 10, size=(8,), endpoint=True))
emit_arr("int_u8", g().integers(5, 250, size=(7,), dtype=np.uint8))
emit_arr("int_u16", g().integers(100, 60000, size=(6,), dtype=np.uint16))
emit_arr("int_u32", g().integers(0, 1_000_000_000, size=(5,), dtype=np.uint32))
emit_arr("int_i16", g().integers(-500, 500, size=(6,), dtype=np.int16))
emit_arr("int_bool", g().integers(0, 2, size=(12,), dtype=bool))
emit_arr("uniform", g().uniform(-4.5, 9.25, size=(6,)))
emit_arr("std_norm", g().standard_normal(size=(2, 4)))
emit_arr("std_norm_f32", g().standard_normal(size=(8,), dtype=np.float32))
emit_arr("normal", g().normal(3.5, 2.25, size=(6,)))
emit_arr("normal_0", g().normal(7.5, 0.0, size=(4,)))
emit_arr("std_exp", g().standard_exponential(size=(6,)))
emit_arr("std_exp_f32", g().standard_exponential(size=(8,), dtype=np.float32))
emit_arr("exp", g().exponential(3.2, size=(6,)))
emit_arr("sg_lt1", g().standard_gamma(0.35, size=(6,)))
emit_arr("sg_eq1", g().standard_gamma(1.0, size=(6,)))
emit_arr("sg_gt1", g().standard_gamma(4.5, size=(6,)))
emit_arr("sg_f32", g().standard_gamma(0.45, size=(6,), dtype=np.float32))
emit_arr("gamma", g().gamma(2.5, 1.8, size=(6,)))
emit_arr("beta1", g().beta(0.3, 0.7, size=(6,)))
emit_arr("beta2", g().beta(2.2, 5.1, size=(6,)))
emit_arr("binom_sm", g().binomial(12, 0.25, size=(6,)))
emit_arr("binom_lg", g().binomial(200, 0.4, size=(6,)))
emit_arr("binom_p0", g().binomial(10, 0.0, size=(4,)))
emit_arr("binom_p1", g().binomial(10, 1.0, size=(4,)))
emit_arr("neg_binom", g().negative_binomial(4.5, 0.35, size=(6,)))
emit_arr("pois_0", g().poisson(0.0, size=(4,)))
emit_arr("pois_sm", g().poisson(2.75, size=(6,)))
emit_arr("pois_lg", g().poisson(35.0, size=(6,)))
emit_arr("geom", g().geometric(0.4, size=(6,)))
emit_arr("hyp_sm", g().hypergeometric(15, 25, 7, size=(6,)))
emit_arr("hyp_lg", g().hypergeometric(60, 80, 35, size=(6,)))
emit_arr("lognorm", g().lognormal(-0.5, 1.1, size=(6,)))
emit_arr("logser", g().logseries(0.72, size=(6,)))
emit_arr("chi2", g().chisquare(6.0, size=(6,)))
emit_arr("nchi_0", g().noncentral_chisquare(5.0, 0.0, size=(6,)))
emit_arr("nchi_lt1", g().noncentral_chisquare(0.6, 2.5, size=(6,)))
emit_arr("nchi_gt1", g().noncentral_chisquare(5.0, 3.0, size=(6,)))
emit_arr("fdist", g().f(5.0, 9.0, size=(6,)))
emit_arr("ncf_0", g().noncentral_f(5.0, 9.0, 0.0, size=(6,)))
emit_arr("ncf", g().noncentral_f(5.0, 9.0, 2.2, size=(6,)))
emit_arr("std_t", g().standard_t(7.0, size=(6,)))
emit_arr("cauchy", g().standard_cauchy(size=(6,)))
emit_arr("laplace", g().laplace(-1.5, 2.0, size=(6,)))
emit_arr("logistic", g().logistic(1.0, 0.8, size=(6,)))
emit_arr("gumbel", g().gumbel(0.5, 1.5, size=(6,)))
emit_arr("rayleigh", g().rayleigh(2.5, size=(6,)))
emit_arr("weibull", g().weibull(1.75, size=(6,)))
emit_arr("pareto", g().pareto(2.5, size=(6,)))
emit_arr("power", g().power(3.0, size=(6,)))
emit_arr("tri", g().triangular(-2.0, 0.5, 5.0, size=(6,)))
emit_arr("tri_l", g().triangular(0.0, 0.0, 3.0, size=(6,)))
emit_arr("vonm_0", g().vonmises(0.5, 0.0, size=(6,)))
emit_arr("vonm_lg", g().vonmises(-0.5, 4.0, size=(6,)))
emit_arr("wald", g().wald(2.0, 3.5, size=(6,)))
emit_arr("zipf", g().zipf(2.4, size=(6,)))
emit_arr("dirichlet", g().dirichlet([0.5, 2.0, 3.5], size=(4,)))
emit_arr("multinom", g().multinomial(20, [0.2, 0.5, 0.3], size=(4,)))
emit_arr("multinom_0", g().multinomial(0, [0.2, 0.8], size=(3,)))
"#,
    );

    let seed = 20261116u64;
    let r = | | default_rng_from_seed(seed).unwrap();

    assert_eq!(r().bytes(13), oracle.arr("bytes").to_u8_vec());
    check_i64_arr(&r().integers(-15, 25, &[3, 4]).unwrap(), &oracle, "int_i64");
    check_i64_arr(&r().integers_inclusive(0, 10, &[8], true).unwrap(), &oracle, "int_incl");
    assert_eq!(r().integers_as::<u8>(5, Some(250), &[7], false).unwrap().to_vec(), oracle.arr("int_u8").to_u8_vec());
    assert_eq!(r().integers_as::<u16>(100, Some(60000), &[6], false).unwrap().to_vec(), oracle.arr("int_u16").to_u16_vec());
    assert_eq!(r().integers_as::<u32>(0, Some(1_000_000_000), &[5], false).unwrap().to_vec(), oracle.arr("int_u32").to_u32_vec());
    assert_eq!(r().integers_as::<i16>(-500, Some(500), &[6], false).unwrap().to_vec(), oracle.arr("int_i16").to_i16_vec());
    assert_eq!(r().integers_as::<bool>(false, Some(true), &[12], true).unwrap().to_vec(), oracle.arr("int_bool").to_bool_vec());

    check_f64_arr(&r().uniform(-4.5, 9.25, &[6]).unwrap(), &oracle, "uniform", 1e-13);
    check_f64_arr(&r().standard_normal(&[2, 4]).unwrap(), &oracle, "std_norm", 1e-13);
    check_f32_arr(&r().standard_normal_f32(&[8]).unwrap(), &oracle, "std_norm_f32", 1e-6);
    check_f64_arr(&r().normal(3.5, 2.25, &[6]).unwrap(), &oracle, "normal", 1e-13);
    check_f64_arr(&r().normal(7.5, 0.0, &[4]).unwrap(), &oracle, "normal_0", 0.0);
    check_f64_arr(&r().standard_exponential(&[6]).unwrap(), &oracle, "std_exp", 1e-13);
    check_f32_arr(&r().standard_exponential_f32(&[8]).unwrap(), &oracle, "std_exp_f32", 1e-6);
    check_f64_arr(&r().exponential(3.2, &[6]).unwrap(), &oracle, "exp", 1e-13);
    check_f64_arr(&r().standard_gamma(0.35, &[6]).unwrap(), &oracle, "sg_lt1", 1e-12);
    check_f64_arr(&r().standard_gamma(1.0, &[6]).unwrap(), &oracle, "sg_eq1", 1e-12);
    check_f64_arr(&r().standard_gamma(4.5, &[6]).unwrap(), &oracle, "sg_gt1", 1e-12);
    check_f32_arr(&r().standard_gamma_f32(0.45, &[6]).unwrap(), &oracle, "sg_f32", 1e-5);
    check_f64_arr(&r().gamma(2.5, 1.8, &[6]).unwrap(), &oracle, "gamma", 1e-12);
    check_f64_arr(&r().beta(0.3, 0.7, &[6]).unwrap(), &oracle, "beta1", 1e-12);
    check_f64_arr(&r().beta(2.2, 5.1, &[6]).unwrap(), &oracle, "beta2", 1e-12);

    check_i64_arr(&r().binomial(12, 0.25, &[6]).unwrap(), &oracle, "binom_sm");
    check_i64_arr(&r().binomial(200, 0.4, &[6]).unwrap(), &oracle, "binom_lg");
    check_i64_arr(&r().binomial(10, 0.0, &[4]).unwrap(), &oracle, "binom_p0");
    check_i64_arr(&r().binomial(10, 1.0, &[4]).unwrap(), &oracle, "binom_p1");
    check_i64_arr(&r().negative_binomial(4.5, 0.35, &[6]).unwrap(), &oracle, "neg_binom");
    check_i64_arr(&r().poisson(0.0, &[4]).unwrap(), &oracle, "pois_0");
    check_i64_arr(&r().poisson(2.75, &[6]).unwrap(), &oracle, "pois_sm");
    check_i64_arr(&r().poisson(35.0, &[6]).unwrap(), &oracle, "pois_lg");
    check_i64_arr(&r().geometric(0.4, &[6]).unwrap(), &oracle, "geom");
    check_i64_arr(&r().hypergeometric(15, 25, 7, &[6]).unwrap(), &oracle, "hyp_sm");
    check_i64_arr(&r().hypergeometric(60, 80, 35, &[6]).unwrap(), &oracle, "hyp_lg");
    check_f64_arr(&r().lognormal(-0.5, 1.1, &[6]).unwrap(), &oracle, "lognorm", 1e-12);
    check_i64_arr(&r().logseries(0.72, &[6]).unwrap(), &oracle, "logser");
    check_f64_arr(&r().chisquare(6.0, &[6]).unwrap(), &oracle, "chi2", 1e-12);
    check_f64_arr(&r().noncentral_chisquare(5.0, 0.0, &[6]).unwrap(), &oracle, "nchi_0", 1e-12);
    check_f64_arr(&r().noncentral_chisquare(0.6, 2.5, &[6]).unwrap(), &oracle, "nchi_lt1", 1e-12);
    check_f64_arr(&r().noncentral_chisquare(5.0, 3.0, &[6]).unwrap(), &oracle, "nchi_gt1", 1e-12);
    check_f64_arr(&r().f(5.0, 9.0, &[6]).unwrap(), &oracle, "fdist", 1e-12);
    check_f64_arr(&r().noncentral_f(5.0, 9.0, 0.0, &[6]).unwrap(), &oracle, "ncf_0", 1e-12);
    check_f64_arr(&r().noncentral_f(5.0, 9.0, 2.2, &[6]).unwrap(), &oracle, "ncf", 1e-12);
    check_f64_arr(&r().standard_t(7.0, &[6]).unwrap(), &oracle, "std_t", 1e-12);
    check_f64_arr(&r().standard_cauchy(&[6]).unwrap(), &oracle, "cauchy", 1e-12);
    check_f64_arr(&r().laplace(-1.5, 2.0, &[6]).unwrap(), &oracle, "laplace", 1e-12);
    check_f64_arr(&r().logistic(1.0, 0.8, &[6]).unwrap(), &oracle, "logistic", 1e-12);
    check_f64_arr(&r().gumbel(0.5, 1.5, &[6]).unwrap(), &oracle, "gumbel", 1e-12);
    check_f64_arr(&r().rayleigh(2.5, &[6]).unwrap(), &oracle, "rayleigh", 1e-12);
    check_f64_arr(&r().weibull(1.75, &[6]).unwrap(), &oracle, "weibull", 1e-12);
    check_f64_arr(&r().pareto(2.5, &[6]).unwrap(), &oracle, "pareto", 1e-12);
    check_f64_arr(&r().power(3.0, &[6]).unwrap(), &oracle, "power", 1e-12);
    check_f64_arr(&r().triangular(-2.0, 0.5, 5.0, &[6]).unwrap(), &oracle, "tri", 1e-12);
    check_f64_arr(&r().triangular(0.0, 0.0, 3.0, &[6]).unwrap(), &oracle, "tri_l", 1e-12);
    check_f64_arr(&r().vonmises(0.5, 0.0, &[6]).unwrap(), &oracle, "vonm_0", 1e-12);
    check_f64_arr(&r().vonmises(-0.5, 4.0, &[6]).unwrap(), &oracle, "vonm_lg", 1e-12);
    check_f64_arr(&r().wald(2.0, 3.5, &[6]).unwrap(), &oracle, "wald", 1e-12);
    check_i64_arr(&r().zipf(2.4, &[6]).unwrap(), &oracle, "zipf");
    check_f64_arr(&r().dirichlet(&[0.5, 2.0, 3.5], Some(&[4])).unwrap(), &oracle, "dirichlet", 1e-12);
    check_i64_arr(&r().multinomial(20, &[0.2, 0.5, 0.3], Some(&[4])).unwrap(), &oracle, "multinom");
    check_i64_arr(&r().multinomial(0, &[0.2, 0.8], Some(&[3])).unwrap(), &oracle, "multinom_0");
}

#[test]
fn test_generator_resampling_broadcasting_multivariate_and_edge_cases() {
    let arr2d = NdArray::from_shape_vec(&[3, 4], (0..12i64).collect()).unwrap();
    let loc = NdArray::from_shape_vec(&[2, 1], vec![10.0, -10.0]).unwrap();
    let scale = NdArray::from_vec(vec![0.5, 2.0, 4.0]);
    let n_arr = NdArray::from_vec(vec![5i64, 50]);
    let p_arr = NdArray::from_shape_vec(&[2, 3], vec![0.1, 0.6, 0.3, 0.5, 0.0, 0.5]).unwrap();
    let cov = NdArray::from_shape_vec(&[2, 2], vec![2.0, 1.2, 1.2, 3.0]).unwrap();

    let mut script = String::new();
    script.push_str(&py_assign_dyn("arr2d", &DynArray::I64(arr2d.clone())));
    script.push('\n');
    script.push_str(&py_assign_f64("loc", &loc));
    script.push('\n');
    script.push_str(&py_assign_f64("scale", &scale));
    script.push('\n');
    script.push_str(&py_assign_dyn("narr", &DynArray::I64(n_arr.clone())));
    script.push('\n');
    script.push_str(&py_assign_f64("parr", &p_arr));
    script.push('\n');
    script.push_str(&py_assign_f64("cov", &cov));
    script.push('\n');
    script.push_str(
        r#"
seed = 777888
def g():
    return np.random.default_rng(seed)

emit_arr("choice_r_t", g().choice(10, size=(2, 3), replace=True))
emit_arr("choice_r_f", g().choice(10, size=(2, 3), replace=False))
emit_arr("choice_p", g().choice(5, size=(8,), replace=True, p=[0.1, 0.2, 0.0, 0.3, 0.4]))
emit_arr("choice_p_nr", g().choice(5, size=(3,), replace=False, p=[0.1, 0.2, 0.0, 0.3, 0.4]))
emit_arr("choice_arr_ax1", g().choice(arr2d, size=(2, 2), replace=False, axis=1))

s_ax0 = arr2d.copy()
g_s0 = g()
g_s0.shuffle(s_ax0, axis=0)
emit_arr("shuf_ax0", s_ax0)

s_ax1 = arr2d.copy()
g_s1 = g()
g_s1.shuffle(s_ax1, axis=1)
emit_arr("shuf_ax1", s_ax1)

emit_arr("perm_r", g().permutation(12))
emit_arr("perm_a1", g().permutation(arr2d, axis=1))
emit_arr("permuted_none", g().permuted(arr2d, axis=None))
emit_arr("permuted_ax1", g().permuted(arr2d, axis=1))

emit_arr("norm_bcast", g().normal(loc, scale))
emit_arr("unif_bcast", g().uniform(np.array([[0.0], [10.0]]), np.array([12.0, 20.0])))
emit_arr("multinom_bcast", g().multinomial(narr, parr))
emit_arr("mvn_chol", g().multivariate_normal([1.0, -2.0], cov, size=(4,), method="cholesky"))

# Test zero-size draw preserves exact RNG state
g_z = g()
_ = g_z.random((0, 5))
emit_arr("after_zero", g_z.random(4))

# Test 0-D scalar draw
emit_arr("scalar0", np.array(g().random()))

# Edge error checks
try:
    g().choice(5, size=(4,), replace=False, p=[0.5, 0.5, 0.0, 0.0, 0.0])
except Exception as e:
    emit_err("choice_few_nz", e)
"#,
    );
    let oracle = run_oracle(&script);
    let seed = 777888u64;
    let r = || default_rng_from_seed(seed).unwrap();

    check_i64_arr(&r().choice_range(10, &[2, 3], true, None, true).unwrap(), &oracle, "choice_r_t");
    check_i64_arr(&r().choice_range(10, &[2, 3], false, None, true).unwrap(), &oracle, "choice_r_f");
    check_i64_arr(
        &r().choice_range(5, &[8], true, Some(&[0.1, 0.2, 0.0, 0.3, 0.4]), true).unwrap(),
        &oracle,
        "choice_p",
    );
    check_i64_arr(
        &r().choice_range(5, &[3], false, Some(&[0.1, 0.2, 0.0, 0.3, 0.4]), true).unwrap(),
        &oracle,
        "choice_p_nr",
    );
    check_i64_arr(&r().choice(&arr2d, &[2, 2], false, None, 1, true).unwrap(), &oracle, "choice_arr_ax1");

    let mut sa0 = arr2d.clone();
    r().shuffle(&mut sa0, 0).unwrap();
    check_i64_arr(&sa0, &oracle, "shuf_ax0");

    let mut sa1 = arr2d.clone();
    r().shuffle(&mut sa1, 1).unwrap();
    check_i64_arr(&sa1, &oracle, "shuf_ax1");

    check_i64_arr(&r().permutation_range(12).unwrap(), &oracle, "perm_r");
    check_i64_arr(&r().permutation(&arr2d, 1).unwrap(), &oracle, "perm_a1");
    check_i64_arr(&r().permuted(&arr2d, None).unwrap(), &oracle, "permuted_none");
    check_i64_arr(&r().permuted(&arr2d, Some(1)).unwrap(), &oracle, "permuted_ax1");

    check_f64_arr(&r().normal_array(&loc, &scale, None).unwrap(), &oracle, "norm_bcast", 1e-13);
    let low_u = NdArray::from_shape_vec(&[2, 1], vec![0.0, 10.0]).unwrap();
    let high_u = NdArray::from_vec(vec![12.0, 20.0]);
    check_f64_arr(&r().uniform_array(&low_u, &high_u, None).unwrap(), &oracle, "unif_bcast", 1e-13);
    check_i64_arr(&r().multinomial_array(&n_arr, &p_arr, None).unwrap(), &oracle, "multinom_bcast");
    check_f64_arr(
        &r().multivariate_normal_with(&[1.0, -2.0], &cov, Some(&[4]), CheckValid::Raise, 1e-8, MvNormalMethod::Cholesky)
            .unwrap(),
        &oracle,
        "mvn_chol",
        1e-13,
    );

    // Zero-size draw preserves RNG state
    let mut gz = r();
    let empty = gz.random(&[0, 5]).unwrap();
    assert_eq!(empty.shape(), &[0, 5]);
    check_f64_arr(&gz.random(&[4]).unwrap(), &oracle, "after_zero", 1e-15);

    // 0-D scalar draw
    let sc0 = r().random(&[]).unwrap();
    assert_eq!(sc0.shape(), &[]);
    check_f64_arr(&sc0, &oracle, "scalar0", 1e-15);

    // Too few non-zero probabilities for replace=false
    if oracle.errors.contains_key("choice_few_nz") {
        assert!(
            r().choice_range(5, &[4], false, Some(&[0.5, 0.5, 0.0, 0.0, 0.0]), true).is_err(),
            "choice with fewer non-zero probs than size should error like NumPy"
        );
    }

    // Invalid parameter error matrix
    let mut g_err = r();
    assert!(g_err.normal(0.0, -1.0, &[2]).is_err());
    assert!(g_err.uniform(5.0, 2.0, &[2]).is_err());
    assert!(g_err.integers(5, 5, &[2]).is_err());
    assert!(g_err.binomial(10, 1.5, &[2]).is_err());
    assert!(g_err.poisson(-0.1, &[2]).is_err());
    assert!(g_err.zipf(1.0, &[2]).is_err());
    assert!(g_err.gamma(-1.0, 1.0, &[2]).is_err());
    assert!(g_err.noncentral_chisquare(1.0, -0.5, &[2]).is_err());
    let non_psd = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap();
    assert!(g_err.multivariate_normal(&[0.0, 0.0], &non_psd, None).is_err());

    // Statistical verification for multivariate_normal (Svd/Eigh) & multivariate_hypergeometric (Count/Marginals)
    let mut g_stat = r();
    let mvn_svd = g_stat
        .multivariate_normal_with(&[1.0, -2.0], &cov, Some(&[10_000]), CheckValid::Raise, 1e-8, MvNormalMethod::Svd)
        .unwrap();
    assert_eq!(mvn_svd.shape(), &[10_000, 2]);
    let svd_data = mvn_svd.to_vec();
    let x0: Vec<f64> = svd_data.iter().step_by(2).copied().collect();
    let x1: Vec<f64> = svd_data.iter().skip(1).step_by(2).copied().collect();
    let (m0, v0) = sample_mean_var(&x0);
    let (m1, v1) = sample_mean_var(&x1);
    assert!((m0 - 1.0).abs() < 0.08 && (m1 - (-2.0)).abs() < 0.08);
    assert!((v0 - 2.0).abs() < 0.15 && (v1 - 3.0).abs() < 0.2);

    let mh = g_stat
        .multivariate_hypergeometric(&[10, 20, 30], 12, Some(&[200]), MvHypergeomMethod::Count)
        .unwrap();
    assert_eq!(mh.shape(), &[200, 3]);
    for row in mh.to_vec().chunks_exact(3) {
        assert_eq!(row.iter().sum::<i64>(), 12);
        assert!(row[0] <= 10 && row[1] <= 20 && row[2] <= 30);
    }

    // Parallel generation > 65_536 threshold: deterministic & statistically valid & 64B aligned
    let mut gp1 = default_rng_from_seed(999u64).unwrap();
    let mut gp2 = default_rng_from_seed(999u64).unwrap();
    let par_u1 = gp1.par_random(&[80_000]).unwrap();
    let par_u2 = gp2.par_random(&[80_000]).unwrap();
    assert_aligned(&par_u1, "par_random");
    assert_eq!(par_u1.to_vec(), par_u2.to_vec(), "par_random must be deterministic");
    let (pm, pv) = sample_mean_var(&par_u1.to_vec());
    assert!((pm - 0.5).abs() < 0.01 && (pv - 1.0 / 12.0).abs() < 0.005);

    let par_n = gp1.par_standard_normal(&[70_000]).unwrap();
    assert_aligned(&par_n, "par_standard_normal");
    let (nm, nv) = sample_mean_var(&par_n.to_vec());
    assert!(nm.abs() < 0.02 && (nv - 1.0).abs() < 0.03);
}

#[test]
fn test_legacy_random_state_against_oracle() {
    let a1 = NdArray::from_vec(vec![10i64, 20, 30, 40, 50, 60]);
    let mut script = py_assign_dyn("a1", &DynArray::I64(a1.clone()));
    script.push_str(
        r#"
seed = 123456
def rs():
    return np.random.RandomState(seed)

emit_arr("rs_sample", rs().random_sample((2, 3)))
emit_arr("rs_rand", rs().rand(2, 3))
emit_arr("rs_randn", rs().randn(2, 3))
emit_arr("rs_randint", rs().randint(-5, 20, size=(8,)).astype(np.int64))
emit_arr("rs_randint_1", rs().randint(10, size=(6,)).astype(np.int64))
emit_arr("rs_rint", rs().random_integers(-3, 3, size=(8,)).astype(np.int64))
emit_arr("rs_tomax", rs().tomaxint((4,)).astype(np.int64))
emit_arr("rs_bytes", np.frombuffer(rs().bytes(11), dtype=np.uint8))
emit_arr("rs_norm", rs().normal(1.5, 2.5, size=(6,)))
emit_arr("rs_unif", rs().uniform(-2.0, 5.0, size=(6,)))
emit_arr("rs_exp", rs().exponential(1.75, size=(6,)))
emit_arr("rs_gam", rs().gamma(0.4, 2.0, size=(6,)))
emit_arr("rs_beta", rs().beta(1.5, 3.0, size=(6,)))
emit_arr("rs_chi2", rs().chisquare(4.0, size=(6,)))
emit_arr("rs_nchi", rs().noncentral_chisquare(3.0, 1.5, size=(6,)))
emit_arr("rs_f", rs().f(4.0, 8.0, size=(6,)))
emit_arr("rs_ncf", rs().noncentral_f(4.0, 8.0, 2.0, size=(6,)))
emit_arr("rs_t", rs().standard_t(5.0, size=(6,)))
emit_arr("rs_cauchy", rs().standard_cauchy(size=(6,)))
emit_arr("rs_laplace", rs().laplace(0.0, 1.5, size=(6,)))
emit_arr("rs_logistic", rs().logistic(-1.0, 2.0, size=(6,)))
emit_arr("rs_gumbel", rs().gumbel(0.2, 1.3, size=(6,)))
emit_arr("rs_lognorm", rs().lognormal(0.1, 0.9, size=(6,)))
emit_arr("rs_rayleigh", rs().rayleigh(3.0, size=(6,)))
emit_arr("rs_weibull", rs().weibull(2.2, size=(6,)))
emit_arr("rs_pareto", rs().pareto(3.0, size=(6,)))
emit_arr("rs_power", rs().power(2.5, size=(6,)))
emit_arr("rs_tri", rs().triangular(-1.0, 0.2, 3.0, size=(6,)))
emit_arr("rs_vonm", rs().vonmises(0.4, 2.0, size=(6,)))
emit_arr("rs_wald", rs().wald(1.5, 2.5, size=(6,)))
emit_arr("rs_binom", rs().binomial(50, 0.35, size=(6,)).astype(np.int64))
emit_arr("rs_nb", rs().negative_binomial(4.0, 0.45, size=(6,)).astype(np.int64))
emit_arr("rs_pois", rs().poisson(15.0, size=(6,)).astype(np.int64))
emit_arr("rs_geom", rs().geometric(0.3, size=(6,)).astype(np.int64))
emit_arr("rs_hyp", rs().hypergeometric(20, 30, 12, size=(6,)).astype(np.int64))
emit_arr("rs_logser", rs().logseries(0.6, size=(6,)).astype(np.int64))
emit_arr("rs_zipf", rs().zipf(2.2, size=(6,)).astype(np.int64))
emit_arr("rs_dir", rs().dirichlet([1.0, 2.0, 3.0], size=(3,)))
emit_arr("rs_mult", rs().multinomial(15, [0.2, 0.3, 0.5], size=(3,)).astype(np.int64))
emit_arr("rs_choice", rs().choice(a1, size=(4,), replace=False).astype(np.int64))
emit_arr("rs_perm_n", rs().permutation(8).astype(np.int64))
s_arr = a1.copy()
rs_s = rs()
rs_s.shuffle(s_arr)
emit_arr("rs_shuffle", s_arr)

rs_key = np.random.RandomState([1, 2, 3, 4])
emit_arr("rs_key_raw", rs_key.random_sample(5))
"#,
    );
    let oracle = run_oracle(&script);
    let seed = 123456u32;
    let rs = || RandomState::from_seed(seed);

    check_f64_arr(&rs().random_sample(&[2, 3]).unwrap(), &oracle, "rs_sample", 1e-15);
    check_f64_arr(&rs().rand(&[2, 3]).unwrap(), &oracle, "rs_rand", 1e-15);
    check_f64_arr(&rs().randn(&[2, 3]).unwrap(), &oracle, "rs_randn", 1e-13);
    check_i64_arr(&rs().randint(-5, Some(20), &[8]).unwrap(), &oracle, "rs_randint");
    check_i64_arr(&rs().randint(10, None, &[6]).unwrap(), &oracle, "rs_randint_1");
    check_i64_arr(&rs().random_integers(-3, Some(3), &[8]).unwrap(), &oracle, "rs_rint");
    check_i64_arr(&rs().tomaxint(&[4]).unwrap(), &oracle, "rs_tomax");
    assert_eq!(rs().bytes(11), oracle.arr("rs_bytes").to_u8_vec());
    check_f64_arr(&rs().normal(1.5, 2.5, &[6]).unwrap(), &oracle, "rs_norm", 1e-13);
    check_f64_arr(&rs().uniform(-2.0, 5.0, &[6]).unwrap(), &oracle, "rs_unif", 1e-13);
    check_f64_arr(&rs().exponential(1.75, &[6]).unwrap(), &oracle, "rs_exp", 1e-13);
    check_f64_arr(&rs().gamma(0.4, 2.0, &[6]).unwrap(), &oracle, "rs_gam", 1e-12);
    check_f64_arr(&rs().beta(1.5, 3.0, &[6]).unwrap(), &oracle, "rs_beta", 1e-12);
    check_f64_arr(&rs().chisquare(4.0, &[6]).unwrap(), &oracle, "rs_chi2", 1e-12);
    check_f64_arr(&rs().noncentral_chisquare(3.0, 1.5, &[6]).unwrap(), &oracle, "rs_nchi", 1e-12);
    check_f64_arr(&rs().f(4.0, 8.0, &[6]).unwrap(), &oracle, "rs_f", 1e-12);
    check_f64_arr(&rs().noncentral_f(4.0, 8.0, 2.0, &[6]).unwrap(), &oracle, "rs_ncf", 1e-12);
    check_f64_arr(&rs().standard_t(5.0, &[6]).unwrap(), &oracle, "rs_t", 1e-12);
    check_f64_arr(&rs().standard_cauchy(&[6]).unwrap(), &oracle, "rs_cauchy", 1e-12);
    check_f64_arr(&rs().laplace(0.0, 1.5, &[6]).unwrap(), &oracle, "rs_laplace", 1e-12);
    check_f64_arr(&rs().logistic(-1.0, 2.0, &[6]).unwrap(), &oracle, "rs_logistic", 1e-12);
    check_f64_arr(&rs().gumbel(0.2, 1.3, &[6]).unwrap(), &oracle, "rs_gumbel", 1e-12);
    check_f64_arr(&rs().lognormal(0.1, 0.9, &[6]).unwrap(), &oracle, "rs_lognorm", 1e-12);
    check_f64_arr(&rs().rayleigh(3.0, &[6]).unwrap(), &oracle, "rs_rayleigh", 1e-12);
    check_f64_arr(&rs().weibull(2.2, &[6]).unwrap(), &oracle, "rs_weibull", 1e-12);
    check_f64_arr(&rs().pareto(3.0, &[6]).unwrap(), &oracle, "rs_pareto", 1e-12);
    check_f64_arr(&rs().power(2.5, &[6]).unwrap(), &oracle, "rs_power", 1e-12);
    check_f64_arr(&rs().triangular(-1.0, 0.2, 3.0, &[6]).unwrap(), &oracle, "rs_tri", 1e-12);
    check_f64_arr(&rs().vonmises(0.4, 2.0, &[6]).unwrap(), &oracle, "rs_vonm", 1e-12);
    check_f64_arr(&rs().wald(1.5, 2.5, &[6]).unwrap(), &oracle, "rs_wald", 1e-12);
    check_i64_arr(&rs().binomial(50, 0.35, &[6]).unwrap(), &oracle, "rs_binom");
    check_i64_arr(&rs().negative_binomial(4.0, 0.45, &[6]).unwrap(), &oracle, "rs_nb");
    check_i64_arr(&rs().poisson(15.0, &[6]).unwrap(), &oracle, "rs_pois");
    check_i64_arr(&rs().geometric(0.3, &[6]).unwrap(), &oracle, "rs_geom");
    check_i64_arr(&rs().hypergeometric(20, 30, 12, &[6]).unwrap(), &oracle, "rs_hyp");
    check_i64_arr(&rs().logseries(0.6, &[6]).unwrap(), &oracle, "rs_logser");
    check_i64_arr(&rs().zipf(2.2, &[6]).unwrap(), &oracle, "rs_zipf");
    check_f64_arr(&rs().dirichlet(&[1.0, 2.0, 3.0], Some(&[3])).unwrap(), &oracle, "rs_dir", 1e-12);
    check_i64_arr(&rs().multinomial(15, &[0.2, 0.3, 0.5], Some(&[3])).unwrap(), &oracle, "rs_mult");
    check_i64_arr(&rs().choice(&a1, &[4], false, None).unwrap(), &oracle, "rs_choice");
    check_i64_arr(&rs().permutation_range(8).unwrap(), &oracle, "rs_perm_n");

    let mut sa = a1.clone();
    rs().shuffle(&mut sa).unwrap();
    check_i64_arr(&sa, &oracle, "rs_shuffle");

    let mut rs_key = RandomState::from_seed_slice(&[1, 2, 3, 4]).unwrap();
    check_f64_arr(&rs_key.random_sample(&[5]).unwrap(), &oracle, "rs_key_raw", 1e-15);

    // State roundtrip including Box-Muller cached second normal (has_gauss=true after 1 draw)
    let mut r_state = rs();
    let _ = r_state.standard_normal(&[1]).unwrap();
    let snap = r_state.get_state();
    assert!(snap.has_gauss);
    let expected = r_state.standard_normal(&[4]).unwrap().to_vec();
    r_state.set_state(snap).unwrap();
    assert_eq!(r_state.standard_normal(&[4]).unwrap().to_vec(), expected);
}

#[test]
fn test_random_edge_cases_empty_dirichlet_mvn_hypergeom_shuffle_and_bitgen_reset() {
    let oracle = run_oracle(
        r#"
rs = np.random.RandomState(0)
g = np.random.default_rng(0)

emit_arr("rs_dir_0", rs.dirichlet([]))
emit_arr("rs_dir_2_3_0", rs.dirichlet([], size=(2, 3)))
emit_arr("g_dir_0", g.dirichlet([]))
emit_arr("g_dir_2_3_0", g.dirichlet([], size=(2, 3)))
emit_arr("g_dir_zero2", g.dirichlet([0.0, 0.0], size=(2,)))
emit_arr("g_dir_zero1", g.dirichlet([0.0, 1.0], size=(2,)))

for name, obj in [("rs", rs), ("g", g)]:
    try:
        obj.multivariate_normal([], np.zeros((0, 0)))
    except Exception as e:
        emit_err(f"{name}_mvn_0", e)

try:
    np.random.RandomState(0).hypergeometric(2**62, 2**62, 1)
except Exception as e:
    emit_err("rs_hyp_ovf", e)

try:
    np.random.default_rng(0).hypergeometric(1_000_000_000, 1, 1)
except Exception as e:
    emit_err("g_hyp_max", e)

rs_reseed = np.random.RandomState(42)
_ = rs_reseed.standard_normal(1)
rs_reseed.seed(99)
emit_arr("rs_after_reseed", rs_reseed.standard_normal(3))

# Shuffle on transposed view via copy/contiguous path
mat = np.arange(6, dtype=np.int64).reshape(2, 3).T.copy()
rs_shuf = np.random.RandomState(77)
rs_shuf.shuffle(mat)
emit_arr("rs_shuf_t", mat)
"#,
    );

    let mut rs = RandomState::from_seed(0);
    let mut g = default_rng_from_seed(0).unwrap();

    check_f64_arr(&rs.dirichlet(&[], None).unwrap(), &oracle, "rs_dir_0", 0.0);
    check_f64_arr(
        &rs.dirichlet(&[], Some(&[2, 3])).unwrap(),
        &oracle,
        "rs_dir_2_3_0",
        0.0,
    );
    check_f64_arr(&g.dirichlet(&[], None).unwrap(), &oracle, "g_dir_0", 0.0);
    check_f64_arr(
        &g.dirichlet(&[], Some(&[2, 3])).unwrap(),
        &oracle,
        "g_dir_2_3_0",
        0.0,
    );
    check_f64_arr(
        &g.dirichlet(&[0.0, 0.0], Some(&[2])).unwrap(),
        &oracle,
        "g_dir_zero2",
        0.0,
    );
    check_f64_arr(
        &g.dirichlet(&[0.0, 1.0], Some(&[2])).unwrap(),
        &oracle,
        "g_dir_zero1",
        0.0,
    );

    assert!(oracle.errors.contains_key("rs_mvn_0"));
    assert!(oracle.errors.contains_key("g_mvn_0"));
    let cov0 = NdArray::<f64>::zeros(&[0, 0]).unwrap();
    assert!(rs.multivariate_normal(&[], &cov0, None).is_err());
    assert!(g.multivariate_normal(&[], &cov0, None).is_err());

    assert!(oracle.errors.contains_key("rs_hyp_ovf"));
    assert!(oracle.errors.contains_key("g_hyp_max"));
    assert!(rs.hypergeometric(1i64 << 62, 1i64 << 62, 1, &[1]).is_err());
    assert!(rs.hypergeometric(1_000_000_000, 1, 1, &[1]).is_err());
    assert!(g.hypergeometric(1_000_000_000, 1, 1, &[1]).is_err());

    // RandomState::bit_generator_mut resets cached Gaussian & Binomial state
    let mut rs_bg = RandomState::from_seed(42);
    let _ = rs_bg.randn(&[1]).unwrap();
    assert!(rs_bg.get_state().has_gauss);
    rs_bg.bit_generator_mut().legacy_seed(99);
    let snap_after = rs_bg.get_state();
    assert!(!snap_after.has_gauss);
    assert_eq!(snap_after.gauss, 0.0);
    check_f64_arr(&rs_bg.randn(&[3]).unwrap(), &oracle, "rs_after_reseed", 1e-13);

    // RandomState::shuffle on non-contiguous transposed NdArray (must not panic)
    let base = NdArray::from_shape_vec(&[2, 3], vec![0i64, 1, 2, 3, 4, 5]).unwrap();
    let mut mat_t = base.t();
    assert!(!mat_t.is_c_contiguous());
    let mut rs_shuf = RandomState::from_seed(77);
    rs_shuf.shuffle(&mut mat_t).unwrap();
    check_i64_arr(&mat_t, &oracle, "rs_shuf_t");
}

#[test]
fn test_round2_domain_b_regressions_against_oracle() {
    let oracle = run_oracle(
        r#"
g_vm_inf = np.random.default_rng(1)
emit_arr("g_vm_inf0", g_vm_inf.vonmises(0.0, np.inf, size=(4,)))
emit_arr("g_vm_inf_next", g_vm_inf.random(2))

emit_arr("g_vm_big", np.random.default_rng(1).vonmises(10.0, 1e7, size=(4,)))
emit_arr("g_vm_negbig", np.random.default_rng(1).vonmises(-10.0, 1e7, size=(4,)))
emit_arr("g_vm_5inf", np.array([np.random.default_rng(1).vonmises(5.0, np.inf), np.random.default_rng(1).vonmises(-5.0, np.inf)]))
emit_arr("g_vm_nank0", np.random.default_rng(1).vonmises(np.nan, 0.0, size=(3,)))
emit_arr("rs_vm_big", np.random.RandomState(1).vonmises(0.0, 1e7, size=(4,)))
emit_arr("rs_vm_nank0", np.random.RandomState(1).vonmises(np.nan, 0.0, size=(3,)))

n_max = (1 << 63) - 1
emit_arr("g_bin_n2_btpe", np.random.default_rng(7).binomial(n_max - 1, 31.0 / (n_max - 1), size=(4,)))
emit_arr("g_bin_max_half", np.random.default_rng(7).binomial(n_max, 0.5, size=(4,)))
emit_arr("g_bin_max_inv", np.random.default_rng(7).binomial(n_max, 10.0 / n_max, size=(4,)))

g_sync = np.random.default_rng(11)
emit_arr("g_bin_n0", g_sync.binomial(0, 0.5, size=(3,)))
emit_arr("g_bin_p0", g_sync.binomial(10, 0.0, size=(3,)))
emit_arr("g_bin_after_zero", g_sync.random(3))
g_sync_p1 = np.random.default_rng(11)
emit_arr("g_bin_p1", g_sync_p1.binomial(10, 1.0, size=(2,)))
emit_arr("g_bin_after_p1", g_sync_p1.random(3))

rs_sync = np.random.RandomState(11)
emit_arr("rs_bin_p0", rs_sync.binomial(10, 0.0, size=(2,)).astype(np.int64))
emit_arr("rs_bin_after_p0", rs_sync.random_sample(3))

try:
    np.random.default_rng(1).negative_binomial(1e19, 0.5)
except Exception as e:
    emit_err("g_nb_max", e)

emit_arr("g_logser0", np.random.default_rng(1).logseries(0.0, size=(4,)))
emit_arr("rs_logser0", np.random.RandomState(1).logseries(0.0, size=(4,)).astype(np.int64))

emit_arr("g_hyp_ns0", np.random.default_rng(1).hypergeometric(10, 10, 0, size=(3,)))
try:
    np.random.RandomState(1).hypergeometric(10, 10, 0, size=(1,))
except Exception as e:
    emit_err("rs_hyp_ns0", e)

emit_arr("g_mvhg_empty_m", np.random.default_rng(1).multivariate_hypergeometric([], 0, size=(2,), method='marginals'))
emit_arr("g_mvhg_empty_c", np.random.default_rng(1).multivariate_hypergeometric([], 0, size=(2,), method='count'))
try:
    np.random.default_rng(1).multivariate_hypergeometric([1_000_000_000, 0], 1, method='marginals')
except Exception as e:
    emit_err("g_mvhg_marg_max", e)

emit_arr("g_ch_empty", np.random.default_rng(1).choice(np.array([], dtype=np.int64), size=(2, 0), replace=False))
emit_arr("rs_ch_empty", np.random.RandomState(1).choice(np.array([], dtype=np.int64), size=(2, 0), replace=True))

try:
    np.random.default_rng(1).permuted(np.array(5.0))
except Exception as e:
    emit_err("g_perm_0d", e)
try:
    np.random.default_rng(1).permuted(np.array(5.0), axis=0)
except Exception as e:
    emit_err("g_perm_0d_ax", e)

emit_arr("g_mvn_nan", np.random.default_rng(1).multivariate_normal([np.nan, 0.0], np.eye(2)))
emit_arr("rs_mvn_nan", np.random.RandomState(1).multivariate_normal([np.nan, 0.0], np.eye(2)))
"#,
    );

    // 1. vonmises wrapped normal (kappa > 1e6) & NaN/Inf edge cases
    let mut g_vm_inf = default_rng_from_seed(1).unwrap();
    check_f64_arr(&g_vm_inf.vonmises(0.0, f64::INFINITY, &[4]).unwrap(), &oracle, "g_vm_inf0", 0.0);
    check_f64_arr(&g_vm_inf.random(&[2]).unwrap(), &oracle, "g_vm_inf_next", 1e-13);

    check_f64_arr(&default_rng_from_seed(1).unwrap().vonmises(10.0, 1e7, &[4]).unwrap(), &oracle, "g_vm_big", 1e-12);
    check_f64_arr(&default_rng_from_seed(1).unwrap().vonmises(-10.0, 1e7, &[4]).unwrap(), &oracle, "g_vm_negbig", 1e-12);
    let vm_5 = default_rng_from_seed(1).unwrap().vonmises(5.0, f64::INFINITY, &[]).unwrap().to_vec()[0];
    let vm_m5 = default_rng_from_seed(1).unwrap().vonmises(-5.0, f64::INFINITY, &[]).unwrap().to_vec()[0];
    assert_f64_close(&[vm_5, vm_m5], &oracle.arr("g_vm_5inf").to_f64_vec(), 1e-12, 1e-12, false, "g_vm_5inf");
    check_f64_arr(&default_rng_from_seed(1).unwrap().vonmises(f64::NAN, 0.0, &[3]).unwrap(), &oracle, "g_vm_nank0", 1e-12);

    let mut g_vm_edge = default_rng_from_seed(1).unwrap();
    assert!(g_vm_edge.vonmises(f64::NAN, 1.0, &[1]).unwrap().to_vec()[0].is_nan());
    assert!(g_vm_edge.vonmises(0.0, f64::NAN, &[1]).unwrap().to_vec()[0].is_nan());
    assert!(g_vm_edge.vonmises(f64::INFINITY, 1.0, &[1]).unwrap().to_vec()[0].is_nan());
    let inf_inf = g_vm_edge.vonmises(f64::INFINITY, f64::INFINITY, &[1]).unwrap().to_vec()[0];
    assert!(inf_inf.is_infinite() && inf_inf.is_sign_positive());
    assert!(g_vm_edge.vonmises(0.0, -1.0, &[1]).is_err());

    check_f64_arr(&RandomState::from_seed(1).vonmises(0.0, 1e7, &[4]).unwrap(), &oracle, "rs_vm_big", 1e-12);
    check_f64_arr(&RandomState::from_seed(1).vonmises(f64::NAN, 0.0, &[3]).unwrap(), &oracle, "rs_vm_nank0", 1e-12);
    let mut rs_vm = RandomState::from_seed(1);
    assert!(rs_vm.vonmises(f64::NAN, 1.0, &[1]).unwrap().to_vec()[0].is_nan());
    assert!(rs_vm.vonmises(0.0, f64::NAN, &[1]).unwrap().to_vec()[0].is_nan());
    assert_eq!(rs_vm.vonmises(0.0, f64::INFINITY, &[1]).unwrap().to_vec()[0], 0.0);
    assert!(rs_vm.vonmises(0.0, -1.0, &[1]).is_err());

    // 2. binomial i64::MAX BTPE / inversion overflow fix (n=i64::MAX succeeds without panic and
    //    matches the non-overflowing BTPE formula at n=i64::MAX-1; NumPy C hits UB in n+1 at i64::MAX)
    check_i64_arr(&default_rng_from_seed(7).unwrap().binomial(i64::MAX, 31.0 / (i64::MAX as f64), &[4]).unwrap(), &oracle, "g_bin_n2_btpe");
    check_i64_arr(&default_rng_from_seed(7).unwrap().binomial(i64::MAX - 1, 31.0 / ((i64::MAX - 1) as f64), &[4]).unwrap(), &oracle, "g_bin_n2_btpe");
    check_i64_arr(&default_rng_from_seed(7).unwrap().binomial(i64::MAX, 0.5, &[4]).unwrap(), &oracle, "g_bin_max_half");
    check_i64_arr(&default_rng_from_seed(7).unwrap().binomial(i64::MAX, 10.0 / (i64::MAX as f64), &[4]).unwrap(), &oracle, "g_bin_max_inv");

    // 3. binomial_impl RNG stream synchronization for n==0 / p==0 vs p==1 / RandomState
    let mut g_sync = default_rng_from_seed(11).unwrap();
    check_i64_arr(&g_sync.binomial(0, 0.5, &[3]).unwrap(), &oracle, "g_bin_n0");
    check_i64_arr(&g_sync.binomial(10, 0.0, &[3]).unwrap(), &oracle, "g_bin_p0");
    check_f64_arr(&g_sync.random(&[3]).unwrap(), &oracle, "g_bin_after_zero", 1e-13);

    let mut g_sync_p1 = default_rng_from_seed(11).unwrap();
    check_i64_arr(&g_sync_p1.binomial(10, 1.0, &[2]).unwrap(), &oracle, "g_bin_p1");
    check_f64_arr(&g_sync_p1.random(&[3]).unwrap(), &oracle, "g_bin_after_p1", 1e-13);

    let mut rs_sync = RandomState::from_seed(11);
    check_i64_arr(&rs_sync.binomial(10, 0.0, &[2]).unwrap(), &oracle, "rs_bin_p0");
    check_f64_arr(&rs_sync.random_sample(&[3]).unwrap(), &oracle, "rs_bin_after_p0", 1e-13);

    // 4. negative_binomial POISSON_LAM_MAX upper bound
    assert!(oracle.errors.contains_key("g_nb_max"));
    assert!(default_rng_from_seed(1).unwrap().negative_binomial(1e19, 0.5, &[1]).is_err());
    assert!(RandomState::from_seed(1).negative_binomial(1e19, 0.5, &[1]).is_err());

    // 5. logseries(0.0)
    check_i64_arr(&default_rng_from_seed(1).unwrap().logseries(0.0, &[4]).unwrap(), &oracle, "g_logser0");
    check_i64_arr(&RandomState::from_seed(1).logseries(0.0, &[4]).unwrap(), &oracle, "rs_logser0");
    assert!(RandomState::from_seed(1).logseries(-0.1, &[1]).is_err());

    // 6. RandomState::hypergeometric rejects nsample < 1, Generator allows nsample == 0
    check_i64_arr(&default_rng_from_seed(1).unwrap().hypergeometric(10, 10, 0, &[3]).unwrap(), &oracle, "g_hyp_ns0");
    assert!(oracle.errors.contains_key("rs_hyp_ns0"));
    assert!(RandomState::from_seed(1).hypergeometric(10, 10, 0, &[1]).is_err());

    // 7. multivariate_hypergeometric empty colors, marginals cap, count OOM guard
    let mut g_mv = default_rng_from_seed(1).unwrap();
    check_i64_arr(&g_mv.multivariate_hypergeometric(&[], 0, Some(&[2]), MvHypergeomMethod::Marginals).unwrap(), &oracle, "g_mvhg_empty_m");
    check_i64_arr(&g_mv.multivariate_hypergeometric(&[], 0, Some(&[2]), MvHypergeomMethod::Count).unwrap(), &oracle, "g_mvhg_empty_c");
    assert!(g_mv.multivariate_hypergeometric(&[], 1, None, MvHypergeomMethod::Marginals).is_err());
    assert!(oracle.errors.contains_key("g_mvhg_marg_max"));
    assert!(g_mv.multivariate_hypergeometric(&[1_000_000_000, 0], 1, None, MvHypergeomMethod::Marginals).is_err());
    assert!(g_mv.multivariate_hypergeometric(&[500_000_000, 500_000_000], 1, None, MvHypergeomMethod::Marginals).is_err());
    assert!(g_mv.multivariate_hypergeometric(&[1_000_000_000_000, 1_000_000_000_000], 1, None, MvHypergeomMethod::Count).is_err());

    // 8. choice with pop_size == 0 and zero-sized output
    let empty_i64 = NdArray::<i64>::from_vec(vec![]);
    check_i64_arr(&default_rng_from_seed(1).unwrap().choice(&empty_i64, &[2, 0], false, None, 0, true).unwrap(), &oracle, "g_ch_empty");
    check_i64_arr(&RandomState::from_seed(1).choice(&empty_i64, &[2, 0], true, None).unwrap(), &oracle, "rs_ch_empty");
    assert!(default_rng_from_seed(1).unwrap().choice_range(0, &[1], false, None, true).is_err());
    assert!(default_rng_from_seed(1).unwrap().choice_range(0, &[0], false, Some(&[]), true).is_err());
    assert!(RandomState::from_seed(1).choice_range(0, &[0], false, Some(&[])).is_err());

    // 9. permuted rejects 0-D scalars
    assert!(oracle.errors.contains_key("g_perm_0d"));
    assert!(oracle.errors.contains_key("g_perm_0d_ax"));
    assert!(default_rng_from_seed(1).unwrap().permuted(&NdArray::from_scalar(5.0), None).is_err());
    assert!(default_rng_from_seed(1).unwrap().permuted(&NdArray::from_scalar(5.0), Some(0)).is_err());

    // 10. multivariate_normal propagates NaN in mean
    let eye2 = NdArray::from_shape_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let g_mvn = default_rng_from_seed(1).unwrap().multivariate_normal(&[f64::NAN, 0.0], &eye2, None).unwrap();
    assert_f64_close(&g_mvn.to_vec(), &oracle.arr("g_mvn_nan").to_f64_vec(), 1e-12, 1e-12, true, "g_mvn_nan");
    let rs_mvn = RandomState::from_seed(1).multivariate_normal(&[f64::NAN, 0.0], &eye2, None).unwrap();
    assert_f64_close(&rs_mvn.to_vec(), &oracle.arr("rs_mvn_nan").to_f64_vec(), 1e-12, 1e-12, true, "rs_mvn_nan");
}

#[test]
fn test_p0_p1_f32_normal_neg_zero_empty_seed_fill_bytes_and_bigint() {
    use numpy_random::SeedInput;

    let oracle = run_oracle(
        r#"
g = np.random.Generator(np.random.PCG64(12345))
n32 = g.standard_normal(200001, dtype=np.float32)
emit_arr("n32_tail", n32)
emit_val("n32_3420", f"{float(n32[3420]):.16g}")

mt_b = np.random.Generator(np.random.MT19937(1)).bytes(7)
emit_val("mt_bytes7", mt_b.hex())

ss_big = np.random.SeedSequence(1234567890123456789012345678901234567890)
emit_arr("ss_big_st", ss_big.generate_state(4))

g0 = np.random.default_rng(0)
emit_arr("g_pois_nz", g0.poisson(-0.0, 3))
emit_arr("g_bin_nz", g0.binomial(5, -0.0, 3))
rs0 = np.random.RandomState(0)
emit_arr("rs_pois_nz", rs0.poisson(-0.0, 3).astype(np.int64))
emit_arr("rs_bin_nz", rs0.binomial(5, -0.0, 3).astype(np.int64))
emit_arr("rs_uni_nz", rs0.uniform(0.0, -0.0, 3))
"#,
    );

    // FIX(F-B3-01): 200,001 f32 standard normals match NumPy bit-for-bit, including tail index 3420
    let mut g = Generator::new(PCG64::from_seed(12345u64).unwrap());
    let n32 = g.standard_normal_f32(&[200_001]).unwrap();
    check_f32_arr(&n32, &oracle, "n32_tail", 0.0);
    let idx3420 = n32.to_vec()[3420];
    assert!(idx3420 < 0.0, "expected negative draw at index 3420, got {idx3420}");

    // FIX(F-B3-02): reject -0.0 on normal/gamma/standard_gamma/rayleigh/weibull/uniform/vonmises, accept on poisson/binomial
    let mut g0 = default_rng_from_seed(0).unwrap();
    assert!(g0.normal(0.0, -0.0, &[3]).is_err());
    assert!(g0.standard_gamma(-0.0, &[3]).is_err());
    assert!(g0.standard_gamma_f32(-0.0, &[3]).is_err());
    assert!(g0.rayleigh(-0.0, &[3]).is_err());
    assert!(g0.weibull(-0.0, &[3]).is_err());
    assert!(g0.uniform(0.0, -0.0, &[3]).is_err());
    assert!(g0.vonmises(0.0, -0.0, &[3]).is_err());
    check_i64_arr(&g0.poisson(-0.0, &[3]).unwrap(), &oracle, "g_pois_nz");
    check_i64_arr(&g0.binomial(5, -0.0, &[3]).unwrap(), &oracle, "g_bin_nz");

    let mut rs0 = RandomState::from_seed(0);
    assert!(rs0.normal(0.0, -0.0, &[3]).is_err());
    assert!(rs0.standard_gamma(-0.0, &[3]).is_err());
    assert!(rs0.rayleigh(-0.0, &[3]).is_err());
    assert!(rs0.weibull(-0.0, &[3]).is_err());
    check_f64_arr(&rs0.uniform(0.0, -0.0, &[3]).unwrap(), &oracle, "rs_uni_nz", 0.0);
    assert!(rs0.vonmises(0.0, -0.0, &[3]).is_err());
    check_i64_arr(&rs0.poisson(-0.0, &[3]).unwrap(), &oracle, "rs_pois_nz");
    check_i64_arr(&rs0.binomial(5, -0.0, &[3]).unwrap(), &oracle, "rs_bin_nz");

    // FIX(F-B3-03): empty seed slices rejected
    assert!(MT19937::from_legacy_key(&[]).is_err());
    assert!(RandomState::from_seed_slice(&[]).is_err());

    // FIX(F-B3-05): fill_bytes agrees with Generator::bytes u32-LE stream
    let mut mt = MT19937::from_seed(1u64).unwrap();
    let mut fb = [0u8; 7];
    mt.fill_bytes(&mut fb);
    let hex: String = fb.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, oracle.val("mt_bytes7"));

    // FIX(F-B3-07): SeedInput arbitrary-precision decimal/hex string coercion
    let si = SeedInput::from_big_uint_str("1234567890123456789012345678901234567890").unwrap();
    let ss_st = SeedSequence::from_seed(si).unwrap().generate_state_u32(4);
    assert_eq!(ss_st, oracle.arr("ss_big_st").to_u32_vec());
}
