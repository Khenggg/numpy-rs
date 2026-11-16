use numpy_random::{
    default_rng, default_rng_from_seed, AnyBitGenerator, BitGenerator, CheckValid, Generator,
    MvHypergeomMethod, MvNormalMethod, RandomState, SeedInput, SeedSequence, MT19937, PCG64,
    PCG64DXSM, Philox, SFC64,
};
use numpy_types::NdArray;
use std::collections::HashSet;

fn rng_deadbeaf() -> Generator<PCG64> {
    let pcg = PCG64::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    Generator::new(pcg)
}

fn rs_deadbeaf() -> RandomState {
    RandomState::from_seed(0xdeadbeaf)
}

fn assert_aligned<T: numpy_types::Scalar>(arr: &NdArray<T>) {
    if let Some(slice) = arr.as_slice() {
        assert_eq!(
            (slice.as_ptr() as usize) % 64,
            0,
            "NdArray pointer {:?} is not 64-byte aligned",
            slice.as_ptr()
        );
    }
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let scale = e.abs().max(1.0);
        assert!(
            diff <= tol * scale,
            "{label}[{i}]: got {a:?}, expected {e:?} (diff={diff:.3e}, tol={:.3e})",
            tol * scale
        );
    }
}

#[test]
fn seed_sequence_spawn_and_any_bit_generator() {
    let mut ss = SeedSequence::from_seed([0x1234_5678_9abc_def0u64, 42]).unwrap();
    let children = ss.spawn(3).unwrap();
    assert_eq!(children.len(), 3);
    assert_eq!(ss.n_children_spawned(), 3);
    let mut states = HashSet::new();
    for c in children {
        let words = c.generate_state_u64(4);
        assert_ne!(words, [0, 0, 0, 0]);
        states.insert(words);
    }
    assert_eq!(states.len(), 3);

    // pool_size < 4 fails cleanly
    assert!(SeedSequence::with_options(SeedInput::Words(vec![1]), Vec::new(), 2, 0).is_err());

    let pcg = PCG64::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    let mut any = AnyBitGenerator::PCG64(pcg);
    assert_eq!(any.next_u64(), 0x60d24054e17a0698);
    assert_eq!(any.next_u64(), 0xd5e79d89856e4f12);

    let spawned = any.spawn(2).unwrap();
    assert_eq!(spawned.len(), 2);

    let mt = MT19937::from_legacy_seed(0xdeadbeaf);
    let mut any_mt = AnyBitGenerator::MT19937(mt);
    let forked = any_mt.spawn_or_fork(2).unwrap();
    assert_eq!(forked.len(), 2);
}

#[test]
fn bit_generators_jump_advance_and_state_roundtrip() {
    // PCG64 jumped(1) vs Python NumPy 2.5.4
    let mut pcg = PCG64::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    let mut pcg_j = pcg.jumped(1);
    assert_eq!(pcg_j.next_u64(), 0xa77dfc1d8f03397f);
    assert_eq!(pcg_j.next_u64(), 0x69cc96387238eaa4);

    // PCG64 state snapshot & advance roundtrip
    let snap = pcg.state();
    let expected = [pcg.next_u64(), pcg.next_u64(), pcg.next_u64()];
    pcg.set_state(snap);
    pcg.advance(2);
    assert_eq!(pcg.next_u64(), expected[2]);

    // PCG64DXSM jumped(1) & advance
    let mut dxsm =
        PCG64DXSM::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    let dx_snap = dxsm.state();
    let d_exp = [dxsm.next_u64(), dxsm.next_u64(), dxsm.next_u64()];
    dxsm.set_state(dx_snap);
    dxsm.advance(2);
    assert_eq!(dxsm.next_u64(), d_exp[2]);
    let mut dx_j = dxsm.jumped(1);
    let _ = dx_j.next_u64();

    // MT19937 jumped(1)
    let mut mt = MT19937::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    let mt_snap = mt.state();
    let mut mt_j = mt.jumped(1);
    assert_ne!(mt_j.next_u32(), mt.next_u32());
    mt.set_state(mt_snap).unwrap();
    assert_eq!(mt.next_u64(), {
        let mut mt2 =
            MT19937::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
        mt2.next_u64()
    });

    // Philox advance & jump
    let mut ph = Philox::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    let ph_snap = ph.state();
    let _ = [ph.next_u64(), ph.next_u64(), ph.next_u64(), ph.next_u64()];
    let after_one_block = ph.next_u64();
    ph.set_state(ph_snap).unwrap();
    ph.advance_u128(1);
    assert_eq!(ph.next_u64(), after_one_block);
    let mut ph_j = ph.jumped(1);
    let _ = ph_j.next_u64();

    // SFC64 state snapshot
    let mut sfc = SFC64::from_seed_seq(SeedSequence::from_seed(0xdeadbeafu64).unwrap()).unwrap();
    let sfc_snap = sfc.state();
    let s0 = sfc.next_u64();
    sfc.set_state(sfc_snap);
    assert_eq!(sfc.next_u64(), s0);
}

#[test]
fn generator_exact_parity_basic_and_ziggurat() {
    let mut g = rng_deadbeaf();
    let r = g.random(&[5]).unwrap();
    assert_aligned(&r);
    assert_close(
        r.as_slice().unwrap(),
        &[
            0.37820818016323443,
            0.83556542022182,
            0.8216032497351045,
            0.9448704222047835,
            0.8428015246147462,
        ],
        1e-15,
        "G_random",
    );

    let mut g = rng_deadbeaf();
    let u = g.uniform(-3.0, 7.0, &[5]).unwrap();
    assert_aligned(&u);
    assert_close(
        u.as_slice().unwrap(),
        &[
            0.7820818016323443,
            5.355654202218201,
            5.216032497351044,
            6.448704222047834,
            5.428015246147462,
        ],
        1e-14,
        "G_uniform",
    );

    let mut g = rng_deadbeaf();
    let sn = g.standard_normal(&[5]).unwrap();
    assert_close(
        sn.as_slice().unwrap(),
        &[
            0.04369918291650199,
            -0.45234849005830907,
            -0.8872645577294029,
            -0.8018403860231904,
            -1.5692271211385624,
        ],
        1e-14,
        "G_standard_normal",
    );

    let mut g = rng_deadbeaf();
    let sn32 = g.standard_normal_f32(&[5]).unwrap();
    let sn32_f64: Vec<f64> = sn32.iter().map(|x| x as f64).collect();
    assert_close(
        &sn32_f64,
        &[
            1.4996371269226074,
            0.4685612916946411,
            -0.34443020820617676,
            -1.3342922925949097,
            -1.3934047222137451,
        ],
        1e-6,
        "G_standard_normal_f32",
    );

    let mut g = rng_deadbeaf();
    let n = g.normal(2.5, 1.5, &[5]).unwrap();
    assert_close(
        n.as_slice().unwrap(),
        &[
            2.565548774374753,
            1.8214772649125364,
            1.1691031634058957,
            1.2972394209652145,
            0.14615931829215612,
        ],
        1e-14,
        "G_normal",
    );

    let mut g = rng_deadbeaf();
    let se = g.standard_exponential(&[5]).unwrap();
    assert_close(
        se.as_slice().unwrap(),
        &[
            1.1966708624783742,
            3.084239990087469,
            3.6809565304491394,
            2.2120164427049813,
            2.1448343293015633,
        ],
        1e-14,
        "G_standard_exponential",
    );

    let mut g = rng_deadbeaf();
    let se32 = g.standard_exponential_f32(&[5]).unwrap();
    let se32_f64: Vec<f64> = se32.iter().map(|x| x as f64).collect();
    assert_close(
        &se32_f64,
        &[
            0.9496188163757324,
            0.2687588930130005,
            0.9307512640953064,
            2.3207879066467285,
            2.438473701477051,
        ],
        1e-6,
        "G_standard_exponential_f32",
    );

    let mut g = rng_deadbeaf();
    let exp = g.exponential(2.5, &[5]).unwrap();
    assert_close(
        exp.as_slice().unwrap(),
        &[
            2.9916771561959354,
            7.710599975218672,
            9.202391326122848,
            5.530041106762454,
            5.362085823253908,
        ],
        1e-14,
        "G_exponential",
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.bytes(10).as_slice(),
        &[152, 6, 122, 225, 84, 64, 210, 96, 18, 79]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.integers_as::<u8>(0, Some(200), &[6], false)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[118, 4, 95, 175, 65, 61]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.integers_as::<u8>(0, Some(200), &[6], true)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[119, 4, 95, 176, 65, 50]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.integers_as::<u8>(0, Some(10), &[5], false)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[5, 0, 4, 8, 3]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.integers(-10, 50, &[6]).unwrap().as_slice().unwrap(),
        &[42, 12, 21, 40, 43, 39]
    );
}

#[test]
fn generator_exact_parity_all_distributions() {
    let mut g = rng_deadbeaf();
    assert_close(
        g.gamma(0.4, 2.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.1759370437467102,
            1.6368649881702946,
            1.870516306139559,
            0.019170533052796472,
            2.375881944357348,
        ],
        1e-12,
        "G_gamma_lt1",
    );

    let mut g = rng_deadbeaf();
    let sg32 = g.standard_gamma_f32(0.4, &[5]).unwrap();
    let sg32_f64: Vec<f64> = sg32.iter().map(|x| x as f64).collect();
    assert_close(
        &sg32_f64,
        &[
            1.2238590717315674,
            0.19612866640090942,
            1.4271811246871948,
            0.06126661226153374,
            0.02800530008971691,
        ],
        1e-6,
        "G_standard_gamma_f32",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.gamma(2.5, 2.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            4.463257523010445,
            2.2109724530846844,
            1.160844147499595,
            3.6916565539502786,
            6.20107290665248,
        ],
        1e-12,
        "G_gamma_gt1",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.beta(0.4, 2.5, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.07370913894316705,
            0.3362923722779352,
            0.004464444195638735,
            0.013852783271924667,
            0.07479850911854843,
        ],
        1e-12,
        "G_beta",
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.binomial(10, 0.2, &[6]).unwrap().as_slice().unwrap(),
        &[2, 3, 3, 4, 3, 2]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.binomial(100, 0.3, &[6]).unwrap().as_slice().unwrap(),
        &[29, 34, 34, 37, 35, 30]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.negative_binomial(5.0, 0.4, &[6])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[7, 7, 7, 5, 5, 6]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.poisson(3.5, &[6]).unwrap().as_slice().unwrap(),
        &[7, 1, 4, 5, 2, 2]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.poisson(25.0, &[6]).unwrap().as_slice().unwrap(),
        &[23, 31, 34, 19, 20, 20]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.geometric(0.35, &[6]).unwrap().as_slice().unwrap(),
        &[2, 5, 5, 7, 5, 2]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.hypergeometric(10, 20, 8, &[6])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[2, 5, 2, 2, 2, 4]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.hypergeometric(50, 60, 30, &[6])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[18, 13, 16, 17, 15, 12]
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.lognormal(0.5, 1.2, &[5]).unwrap().as_slice().unwrap(),
        &[
            1.737485615371744,
            0.9580855656187725,
            0.5685207486481199,
            0.6298910171397871,
            0.2508067528450691,
        ],
        1e-12,
        "G_lognormal",
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.logseries(0.65, &[6]).unwrap().as_slice().unwrap(),
        &[2, 1, 1, 1, 2, 2]
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.chisquare(5.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            4.463257523010445,
            2.2109724530846844,
            1.160844147499595,
            3.6916565539502786,
            6.20107290665248,
        ],
        1e-12,
        "G_chisquare",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.noncentral_chisquare(5.0, 2.0, &[5])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[
            3.7251172590901844,
            1.8097041008603179,
            6.673538841064985,
            8.209224576644491,
            1.3662853470265286,
        ],
        1e-12,
        "G_noncentral_chi",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.f(4.0, 7.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            1.6432998299235,
            0.22118718508961843,
            1.3970454073278036,
            0.4164498059315636,
            0.20699553621283862,
        ],
        1e-12,
        "G_f",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.noncentral_f(4.0, 7.0, 2.5, &[5])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[
            1.31043955382996,
            1.3553976747863734,
            0.2621245536457957,
            0.47717036445560873,
            1.4249002148648406,
        ],
        1e-12,
        "G_noncentral_f",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.standard_t(5.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.055201348177545496,
            -1.664125097668624,
            -0.2971000528340793,
            0.30199824944446535,
            -2.884758893021882,
        ],
        1e-12,
        "G_standard_t",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.standard_cauchy(&[5]).unwrap().as_slice().unwrap(),
        &[
            -0.09660512608513191,
            1.1065351324219057,
            0.8725286680877988,
            0.44671555922404704,
            2.0066146788154837,
        ],
        1e-12,
        "G_standard_cauchy",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.laplace(1.0, 2.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.4416733735354281,
            3.2241905972699683,
            3.0611961888922896,
            5.409843448397962,
            3.31419783408241,
        ],
        1e-12,
        "G_laplace",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.logistic(-1.0, 0.5, &[5]).unwrap().as_slice().unwrap(),
        &[
            -1.2485802783651363,
            -0.1872020770663464,
            -0.23637619499786622,
            0.4206807122543976,
            -0.1603888452324963,
        ],
        1e-12,
        "G_logistic",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.gumbel(0.5, 1.5, &[5]).unwrap().as_slice().unwrap(),
        &[
            1.6161873016761987,
            -0.38604238039972893,
            -0.31674911351259927,
            -1.0960669304128379,
            -0.4229779839601706,
        ],
        1e-12,
        "G_gumbel",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.rayleigh(2.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            3.094085793869813,
            4.967284964716616,
            5.426569104286162,
            4.2066770189354745,
            4.142303059218689,
        ],
        1e-12,
        "G_rayleigh",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.weibull(2.5, &[5]).unwrap().as_slice().unwrap(),
        &[
            1.0744590955625424,
            1.5691312279199099,
            1.6841635954833554,
            1.373774150849471,
            1.3569301768232018,
        ],
        1e-12,
        "G_weibull",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.pareto(3.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.4901701193386492,
            1.7956929386808524,
            2.410931803580478,
            1.0903652557727685,
            1.044073798097057,
        ],
        1e-12,
        "G_pareto",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.power(2.5, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.8659498366350314,
            0.9814364258362577,
            0.9898432170277581,
            0.954679366268615,
            0.9514086142476982,
        ],
        1e-12,
        "G_power",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.triangular(-1.0, 0.5, 3.0, &[5])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[
            0.506424615463239,
            1.7176795261005149,
            1.6643475367263552,
            2.2575070519175515,
            1.7462118385259264,
        ],
        1e-12,
        "G_triangular",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.vonmises(0.5, 3.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.8838734471269731,
            -0.008419953166920013,
            0.6434693933027069,
            2.083424145696206,
            0.6768086088459815,
        ],
        1e-12,
        "G_vonmises",
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.wald(2.0, 3.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            2.072644900975777,
            4.065846133767507,
            0.5983600480910736,
            1.6584187752518265,
            1.2706610359715442,
        ],
        1e-12,
        "G_wald",
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.zipf(2.0, &[6]).unwrap().as_slice().unwrap(),
        &[1, 6, 14, 1, 1, 1]
    );

    let mut g = rng_deadbeaf();
    assert_close(
        g.dirichlet(&[0.05, 1.5, 3.0], Some(&[2]))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[
            2.792699770778051e-09,
            0.34793097300066633,
            0.6520690242066339,
            3.270553160284794e-17,
            0.5256413157857274,
            0.4743586842142725,
        ],
        1e-12,
        "G_dirichlet",
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.multinomial(20, &[0.2, 0.3, 0.5], Some(&[3]))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[3, 8, 9, 6, 8, 6, 6, 5, 9]
    );
}

#[test]
fn generator_multivariate_choice_permutation_shuffle_parity() {
    let cov = NdArray::from_shape_vec(&[2, 2], vec![2.0, 0.8, 0.8, 1.5]).unwrap();
    let mean = [1.0, -1.0];

    let mut g = rng_deadbeaf();
    let mvn_chol = g
        .multivariate_normal_with(
            &mean,
            &cov,
            Some(&[3]),
            CheckValid::Raise,
            1e-8,
            MvNormalMethod::Cholesky,
        )
        .unwrap();
    assert_aligned(&mvn_chol);
    assert_close(
        mvn_chol.as_slice().unwrap(),
        &[
            1.0617999771451399,
            -1.4666562444448714,
            -0.25478157095388765,
            -2.3729342386164713,
            -1.2192222771578427,
            -3.8413404504880755,
        ],
        1e-12,
        "G_mvn_chol",
    );

    // SVD and Eigh produce valid samples matching shape and covariance empirically
    let mut g = rng_deadbeaf();
    let mvn_svd = g
        .multivariate_normal_with(
            &mean,
            &cov,
            Some(&[10_000]),
            CheckValid::Raise,
            1e-8,
            MvNormalMethod::Svd,
        )
        .unwrap();
    assert_eq!(mvn_svd.shape(), &[10_000, 2]);
    let svd_slice = mvn_svd.as_slice().unwrap();
    let mut m0 = 0.0;
    let mut m1 = 0.0;
    for chunk in svd_slice.chunks_exact(2) {
        m0 += chunk[0];
        m1 += chunk[1];
    }
    m0 /= 10_000.0;
    m1 /= 10_000.0;
    assert!((m0 - 1.0).abs() < 0.06, "svd mean0={m0}");
    assert!((m1 + 1.0).abs() < 0.06, "svd mean1={m1}");
    let mut c01 = 0.0;
    for chunk in svd_slice.chunks_exact(2) {
        c01 += (chunk[0] - m0) * (chunk[1] - m1);
    }
    c01 /= 10_000.0;
    assert!((c01 - 0.8).abs() < 0.08, "svd cov01={c01}");

    let mut g = rng_deadbeaf();
    let mvn_eigh = g
        .multivariate_normal_with(
            &mean,
            &cov,
            Some(&[5_000]),
            CheckValid::Raise,
            1e-8,
            MvNormalMethod::Eigh,
        )
        .unwrap();
    assert_eq!(mvn_eigh.shape(), &[5_000, 2]);

    // multivariate_hypergeometric
    let mut g = rng_deadbeaf();
    assert_eq!(
        g.multivariate_hypergeometric(
            &[10, 15, 20],
            12,
            Some(&[3]),
            MvHypergeomMethod::Marginals,
        )
        .unwrap()
        .as_slice()
        .unwrap(),
        &[3, 5, 4, 1, 7, 4, 2, 5, 5]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.multivariate_hypergeometric(&[10, 15, 20], 12, Some(&[3]), MvHypergeomMethod::Count)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[2, 8, 2, 4, 4, 4, 1, 4, 7]
    );

    // choice & permutation & shuffle & permuted
    let mut g = rng_deadbeaf();
    assert_eq!(
        g.choice_range(10, &[5], true, None, true)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[8, 3, 5, 8, 8]
    );
    let mut g = rng_deadbeaf();
    assert_eq!(
        g.choice_range(10, &[5], false, None, true)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[7, 5, 4, 2, 8]
    );
    let mut g = rng_deadbeaf();
    assert_eq!(
        g.choice_range(5, &[5], true, Some(&[0.1, 0.2, 0.3, 0.25, 0.15]), true)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[2, 3, 3, 4, 3]
    );
    let mut g = rng_deadbeaf();
    assert_eq!(
        g.choice_range(5, &[3], false, Some(&[0.1, 0.2, 0.3, 0.25, 0.15]), true)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[2, 3, 4]
    );

    let mut g = rng_deadbeaf();
    assert_eq!(
        g.permutation_range(10).unwrap().as_slice().unwrap(),
        &[0, 3, 9, 5, 6, 7, 1, 2, 4, 8]
    );

    let mut g = rng_deadbeaf();
    let mut arr = NdArray::from_vec((0..10i64).collect());
    g.shuffle(&mut arr, 0).unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[0, 3, 9, 5, 6, 7, 1, 2, 4, 8]);

    let mut g = rng_deadbeaf();
    let a2 = NdArray::from_shape_vec(&[3, 4], (0..12i64).collect()).unwrap();
    let permuted = g.permuted(&a2, Some(0)).unwrap();
    assert_eq!(
        permuted.as_slice().unwrap(),
        &[4, 1, 2, 11, 8, 5, 6, 3, 0, 9, 10, 7]
    );

    // Generator::choice on NdArray along axis
    let mut g = rng_deadbeaf();
    let picked = g.choice(&a2, &[2], false, None, 0, true).unwrap();
    assert_eq!(picked.shape(), &[2, 4]);
}

#[test]
fn random_state_exact_parity_with_numpy() {
    let mut rs = rs_deadbeaf();
    assert_close(
        rs.rand(&[4]).unwrap().as_slice().unwrap(),
        &[
            0.7906980309529287,
            0.7545095023171725,
            0.6715417379758448,
            0.6961951052087804,
        ],
        1e-15,
        "RS_rand",
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.randn(&[4]).unwrap().as_slice().unwrap(),
        &[
            0.6689419810625815,
            0.764058374819756,
            1.2153557483336528,
            1.062637302323269,
        ],
        1e-14,
        "RS_randn",
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.randint(0, Some(10), &[4]).unwrap().as_slice().unwrap(),
        &[1, 4, 6, 6]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.randint(-10, Some(50), &[6]).unwrap().as_slice().unwrap(),
        &[39, 42, 3, 4, 5, 12]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.bytes(10).as_slice(),
        &[177, 47, 107, 202, 116, 169, 162, 63, 254, 136]
    );

    let mut rs_arr = RandomState::from_seed_slice(&[1, 2, 3]).unwrap();
    assert_close(
        rs_arr.rand(&[3]).unwrap().as_slice().unwrap(),
        &[0.6098612722867289, 0.8866970434146851, 0.47633673750662764],
        1e-15,
        "RS_seed_slice",
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.binomial(100, 0.3, &[6]).unwrap().as_slice().unwrap(),
        &[34, 33, 32, 32, 33, 30]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.binomial(10, 0.2, &[6]).unwrap().as_slice().unwrap(),
        &[3, 3, 2, 3, 3, 2]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.poisson(3.5, &[6]).unwrap().as_slice().unwrap(),
        &[7, 5, 5, 3, 1, 4]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.poisson(25.0, &[6]).unwrap().as_slice().unwrap(),
        &[28, 29, 22, 23, 28, 26]
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.gamma(0.4, 2.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            1.368066464409705,
            0.7593108805625046,
            1.2547277229527607,
            0.17126307896929568,
            0.9547536652690148,
        ],
        1e-12,
        "RS_gamma_lt1",
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.gamma(2.5, 2.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            6.616031287281526,
            6.994296978614834,
            4.501779121015333,
            10.659937032840132,
            1.397172786946598,
        ],
        1e-12,
        "RS_gamma_gt1",
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.beta(0.4, 2.5, &[5]).unwrap().as_slice().unwrap(),
        &[
            0.13212446314066356,
            0.042975519443119066,
            0.0215005707049409,
            0.0014642847801280246,
            0.6895854737418023,
        ],
        1e-12,
        "RS_beta",
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.hypergeometric(10, 20, 8, &[6])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[3, 3, 2, 1, 4, 2]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.hypergeometric(50, 60, 30, &[6])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[15, 15, 14, 12, 15, 15]
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.wald(2.0, 3.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            3.430702557316653,
            3.6961826711348684,
            1.909851361186979,
            0.6083305185939858,
            0.6792397274557154,
        ],
        1e-12,
        "RS_wald",
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.vonmises(0.5, 3.0, &[5]).unwrap().as_slice().unwrap(),
        &[
            1.4388064144864572,
            -0.07724345122196397,
            0.8787543876094501,
            0.6130831235060028,
            -0.16449477271884483,
        ],
        1e-12,
        "RS_vonmises",
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.multinomial(20, &[0.2, 0.3, 0.5], Some(&[3]))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[5, 7, 8, 5, 7, 8, 5, 6, 9]
    );

    let mut rs = rs_deadbeaf();
    assert_close(
        rs.dirichlet(&[0.5, 1.5, 3.0], Some(&[2]))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[
            0.10041463315126645,
            0.3481020789450045,
            0.5514832879037289,
            0.020073329274428644,
            0.36292492351383715,
            0.6170017472117341,
        ],
        1e-12,
        "RS_dirichlet",
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.permutation_range(10).unwrap().as_slice().unwrap(),
        &[3, 0, 7, 9, 2, 8, 5, 6, 4, 1]
    );

    let mut rs = rs_deadbeaf();
    assert_eq!(
        rs.choice_range(10, &[5], false, None)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[3, 0, 7, 9, 2]
    );
}

#[test]
fn statistical_sanity_alignment_broadcasting_and_parallel_streams() {
    let mut g = default_rng_from_seed(12345u64).unwrap();

    // 0-D scalar & empty shapes
    let scalar = g.random(&[]).unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.len(), 1);
    assert_aligned(&scalar);

    let empty = g.normal(0.0, 1.0, &[2, 0, 3]).unwrap();
    assert_eq!(empty.shape(), &[2, 0, 3]);
    assert_eq!(empty.len(), 0);

    // Parallel Rayon streams (> DEFAULT_PARALLEL_THRESHOLD = 65_536)
    let n_par = 200_000usize;
    let par_u = g.par_random(&[n_par]).unwrap();
    assert_aligned(&par_u);
    assert_eq!(par_u.len(), n_par);
    let u_sum: f64 = par_u.iter().sum();
    let u_mean = u_sum / n_par as f64;
    assert!((u_mean - 0.5).abs() < 0.01, "par_random mean={u_mean}");

    let par_n = g.par_standard_normal(&[n_par]).unwrap();
    assert_aligned(&par_n);
    let n_mean = par_n.iter().sum::<f64>() / n_par as f64;
    let n_var = par_n.iter().map(|x| (x - n_mean).powi(2)).sum::<f64>() / n_par as f64;
    assert!(n_mean.abs() < 0.02, "par_normal mean={n_mean}");
    assert!((n_var - 1.0).abs() < 0.03, "par_normal var={n_var}");

    let par_e = g.par_standard_exponential(&[n_par]).unwrap();
    assert_aligned(&par_e);
    let e_mean = par_e.iter().sum::<f64>() / n_par as f64;
    assert!((e_mean - 1.0).abs() < 0.02, "par_exp mean={e_mean}");

    let par_g = g.par_gamma(3.0, 2.0, &[n_par]).unwrap();
    assert_aligned(&par_g);
    let g_mean = par_g.iter().sum::<f64>() / n_par as f64;
    assert!((g_mean - 6.0).abs() < 0.1, "par_gamma mean={g_mean}");

    let par_i = g.par_integers(-5, 5, &[n_par]).unwrap();
    assert_aligned(&par_i);
    assert!(par_i.iter().all(|v| (-5..5).contains(&v)));

    let par_u32 = g.par_random_f32(&[n_par]).unwrap();
    assert_aligned(&par_u32);
    assert!(par_u32.iter().all(|v| (0.0..1.0).contains(&v)));

    let par_n32 = g.par_standard_normal_f32(&[n_par]).unwrap();
    assert_aligned(&par_n32);

    let par_e32 = g.par_standard_exponential_f32(&[n_par]).unwrap();
    assert_aligned(&par_e32);

    let par_nf64 = g.par_normal(10.0, 2.0, &[n_par]).unwrap();
    assert_aligned(&par_nf64);

    let par_uf64 = g.par_uniform(-2.0, 4.0, &[n_par]).unwrap();
    assert_aligned(&par_uf64);
    assert!(par_uf64.iter().all(|v| (-2.0..4.0).contains(&v)));

    let par_ef64 = g.par_exponential(0.5, &[n_par]).unwrap();
    assert_aligned(&par_ef64);

    // Array parameter broadcasting
    let locs = NdArray::from_shape_vec(&[2, 1], vec![0.0, 100.0]).unwrap();
    let scales = NdArray::from_shape_vec(&[1, 3], vec![0.1, 0.1, 0.1]).unwrap();
    let b_norm = g.normal_array(&locs, &scales, None).unwrap();
    assert_eq!(b_norm.shape(), &[2, 3]);
    assert_aligned(&b_norm);
    let s = b_norm.as_slice().unwrap();
    assert!(s[0].abs() < 5.0 && s[1].abs() < 5.0 && s[2].abs() < 5.0);
    assert!((s[3] - 100.0).abs() < 5.0 && (s[4] - 100.0).abs() < 5.0);

    // bool & u64 bounded integers
    let bools = g.integers_as::<bool>(false, Some(true), &[128], true).unwrap();
    assert_eq!(bools.len(), 128);
    assert!(bools.iter().any(|b| b) && bools.iter().any(|b| !b));

    let u64_closed = g
        .integers_as::<u64>(0, Some(u64::MAX), &[16], true)
        .unwrap();
    assert_eq!(u64_closed.len(), 16);

    // Error handling checks
    assert!(g.normal(0.0, -1.0, &[3]).is_err());
    assert!(g.uniform(5.0, 1.0, &[3]).is_err());
    assert!(g.beta(0.0, 1.0, &[3]).is_err());
    assert!(g.binomial(10, 1.5, &[3]).is_err());
    assert!(g.poisson(-1.0, &[3]).is_err());
    assert!(g.geometric(0.0, &[3]).is_err());
    assert!(g.hypergeometric(5, 5, 11, &[3]).is_err());
    assert!(g.logseries(1.0, &[3]).is_err());
    assert!(g.zipf(1.0, &[3]).is_err());
    assert!(g.triangular(2.0, 1.0, 3.0, &[3]).is_err());
    assert!(g.choice_range(5, &[6], false, None, true).is_err());
    let non_psd = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap();
    assert!(g
        .multivariate_normal_with(
            &[0.0, 0.0],
            &non_psd,
            Some(&[2]),
            CheckValid::Raise,
            1e-8,
            MvNormalMethod::Svd,
        )
        .is_err());
    assert!(g
        .multivariate_normal_with(
            &[0.0, 0.0],
            &non_psd,
            Some(&[2]),
            CheckValid::Raise,
            1e-8,
            MvNormalMethod::Cholesky,
        )
        .is_err());
    let _ = default_rng();
}

#[test]
fn par_streams_are_invariant_to_thread_pool_size() {
    // FIX(F-B3-04): par_random / par_normal outputs must be independent of rayon worker count.
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let (r1, n1) = pool1.install(|| {
        let mut g = default_rng_from_seed(777u64).unwrap();
        (
            g.par_random(&[70_000]).unwrap(),
            g.par_normal(1.0, 2.0, &[70_000]).unwrap(),
        )
    });
    let (r4, n4) = pool4.install(|| {
        let mut g = default_rng_from_seed(777u64).unwrap();
        (
            g.par_random(&[70_000]).unwrap(),
            g.par_normal(1.0, 2.0, &[70_000]).unwrap(),
        )
    });
    assert_eq!(r1.to_vec(), r4.to_vec());
    assert_eq!(n1.to_vec(), n4.to_vec());
}
