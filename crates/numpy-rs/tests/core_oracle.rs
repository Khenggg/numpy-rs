#![cfg(feature = "core")]

mod common;

use common::{assert_aligned, assert_dyn_matches_oracle, non_contiguous_2d_views, py_assign_dyn, run_oracle};
use numpy_core::{
    append, arange, argwhere, array_split, atleast_1d, atleast_2d, atleast_3d, boolean_mask,
    choose, column_stack, compress, concatenate, copyto, delete, diag, diagflat, diagonal, dsplit,
    dstack, dyn_arange, dyn_einsum, dyn_eye, dyn_full, dyn_geomspace, dyn_linspace, dyn_logspace,
    dyn_meshgrid, dyn_ones, dyn_zeros, einsum, einsum_with_optimize, expand_dims, extract, eye,
    fill_diagonal, flatnonzero, flatten, flip, fliplr, flipud, full, geomspace, hsplit, hstack,
    identity, index_advanced, index_fancy, indices, insert, linspace, linspace_retstep, logspace,
    may_share_memory, meshgrid, moveaxis, nonzero, ones, pad, put, put_along_axis, putmask, ravel,
    ravel_multi_index, repeat, reshape, resize, roll, roll_axes, rollaxis, rot90, select,
    shares_memory, split, squeeze, stack, swapaxes, take, take_along_axis, tile, transpose, tri,
    tril, trim_zeros, triu, unravel_index, unstack, vander, vsplit, vstack, where_select, zeros,
    DeleteObj, DynScalar, EinsumOptimize, FancyArg, IndexMode, InsertObj, MeshgridIndexing,
    PadMode, PadWidth, ReflectType, SplitSpec, TrimDirection,
};
use numpy_types::{f16, Complex64, DType, DynArray, NdArray, Order, Slice, SliceArg};

#[test]
fn creation_ranges_grids_and_huge_alloc_against_numpy() {
    let py = r#"
emit_arr("zeros", np.zeros((2, 3), dtype=np.float64))
emit_arr("ones", np.ones((2, 3), dtype=np.int32))
emit_arr("full", np.full((2, 2), -7.5, dtype=np.float64))
emit_arr("arange_pos", np.arange(1.0, 6.5, 1.5, dtype=np.float64))
emit_arr("arange_neg", np.arange(10, 1, -3, dtype=np.int64))
emit_arr("linspace_ep", np.linspace(-2.0, 2.0, 5, endpoint=True, dtype=np.float64))
emit_arr("linspace_noep", np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64))
emit_arr("logspace", np.logspace(0.0, 3.0, 4, endpoint=True, base=2.0, dtype=np.float64))
emit_arr("geomspace", np.geomspace(1.0, 16.0, 5, endpoint=True, dtype=np.float64))
emit_arr("eye", np.eye(3, 4, k=1, dtype=np.float64))
emit_arr("identity", np.identity(3, dtype=np.int64))
emit_arr("diag_vec", np.diag(np.array([1.0, 2.0, 3.0]), k=-1))
emit_arr("diag_mat", np.diag(np.arange(12.0).reshape(3, 4), k=1))
emit_arr("diagflat", np.diagflat(np.array([[1, 2], [3, 4]]), k=0))
emit_arr("tri", np.tri(3, 4, k=1, dtype=np.float64))
emit_arr("tril", np.tril(np.arange(12.0).reshape(3, 4), k=0))
emit_arr("triu", np.triu(np.arange(12.0).reshape(3, 4), k=1))
emit_arr("vander_dec", np.vander(np.array([1.0, 2.0, 3.0]), N=4, increasing=False))
emit_arr("vander_inc", np.vander(np.array([1.0, 2.0, 3.0]), N=3, increasing=True))
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0])
mx, my = np.meshgrid(x, y, indexing='xy')
emit_arr("mesh_xy_x", mx)
emit_arr("mesh_xy_y", my)
mix, miy = np.meshgrid(x, y, indexing='ij', sparse=True)
emit_arr("mesh_ij_sp_x", mix)
emit_arr("mesh_ij_sp_y", miy)
emit_arr("indices", np.indices((2, 3), dtype=np.int64))
"#;
    let oracle = run_oracle(py);

    assert_dyn_matches_oracle(
        &zeros::<f64>(&[2, 3]).unwrap().into(),
        oracle.arr("zeros"),
        0.0,
        0.0,
        false,
        false,
        "zeros",
    );
    assert_dyn_matches_oracle(
        &ones::<i32>(&[2, 3]).unwrap().into(),
        oracle.arr("ones"),
        0.0,
        0.0,
        false,
        false,
        "ones",
    );
    assert_dyn_matches_oracle(
        &full(&[2, 2], -7.5f64).unwrap().into(),
        oracle.arr("full"),
        0.0,
        0.0,
        false,
        false,
        "full",
    );
    assert_dyn_matches_oracle(
        &arange(1.0f64, 6.5, 1.5).unwrap().into(),
        oracle.arr("arange_pos"),
        1e-14,
        1e-14,
        false,
        false,
        "arange_pos",
    );
    assert_dyn_matches_oracle(
        &arange(10i64, 1, -3).unwrap().into(),
        oracle.arr("arange_neg"),
        0.0,
        0.0,
        false,
        false,
        "arange_neg",
    );
    let (ls, step) = linspace_retstep(-2.0f64, 2.0, 5, true).unwrap();
    assert!((step - 1.0).abs() < 1e-14);
    assert_dyn_matches_oracle(
        &ls.into(),
        oracle.arr("linspace_ep"),
        1e-14,
        1e-14,
        false,
        false,
        "linspace_ep",
    );
    assert_dyn_matches_oracle(
        &linspace(0.0f64, 1.0, 4, false).unwrap().into(),
        oracle.arr("linspace_noep"),
        1e-14,
        1e-14,
        false,
        false,
        "linspace_noep",
    );
    assert_dyn_matches_oracle(
        &logspace(0.0f64, 3.0, 4, true, 2.0).unwrap().into(),
        oracle.arr("logspace"),
        1e-13,
        1e-13,
        false,
        false,
        "logspace",
    );
    assert_dyn_matches_oracle(
        &geomspace(1.0f64, 16.0, 5, true).unwrap().into(),
        oracle.arr("geomspace"),
        1e-13,
        1e-13,
        false,
        false,
        "geomspace",
    );
    assert_dyn_matches_oracle(
        &eye::<f64>(3, Some(4), 1).unwrap().into(),
        oracle.arr("eye"),
        0.0,
        0.0,
        false,
        false,
        "eye",
    );
    assert_dyn_matches_oracle(
        &identity::<i64>(3).unwrap().into(),
        oracle.arr("identity"),
        0.0,
        0.0,
        false,
        false,
        "identity",
    );

    let v3 = NdArray::from_vec(vec![1.0f64, 2.0, 3.0]);
    let m34 = arange(0.0f64, 12.0, 1.0).unwrap().reshape(&[3, 4]).unwrap();
    assert_dyn_matches_oracle(
        &diag(&v3, -1).unwrap().into(),
        oracle.arr("diag_vec"),
        0.0,
        0.0,
        false,
        false,
        "diag_vec",
    );
    assert_dyn_matches_oracle(
        &diag(&m34, 1).unwrap().into(),
        oracle.arr("diag_mat"),
        0.0,
        0.0,
        false,
        false,
        "diag_mat",
    );
    let m22 = NdArray::from_shape_vec(&[2, 2], vec![1i64, 2, 3, 4]).unwrap();
    assert_dyn_matches_oracle(
        &diagflat(&m22, 0).unwrap().into(),
        oracle.arr("diagflat"),
        0.0,
        0.0,
        false,
        false,
        "diagflat",
    );
    assert_dyn_matches_oracle(
        &tri::<f64>(3, Some(4), 1).unwrap().into(),
        oracle.arr("tri"),
        0.0,
        0.0,
        false,
        false,
        "tri",
    );
    assert_dyn_matches_oracle(
        &tril(&m34, 0).unwrap().into(),
        oracle.arr("tril"),
        0.0,
        0.0,
        false,
        false,
        "tril",
    );
    assert_dyn_matches_oracle(
        &triu(&m34, 1).unwrap().into(),
        oracle.arr("triu"),
        0.0,
        0.0,
        false,
        false,
        "triu",
    );
    assert_dyn_matches_oracle(
        &vander(&v3, Some(4), false).unwrap().into(),
        oracle.arr("vander_dec"),
        1e-14,
        1e-14,
        false,
        false,
        "vander_dec",
    );
    assert_dyn_matches_oracle(
        &vander(&v3, Some(3), true).unwrap().into(),
        oracle.arr("vander_inc"),
        1e-14,
        1e-14,
        false,
        false,
        "vander_inc",
    );

    let y = NdArray::from_vec(vec![10.0f64, 20.0]);
    let mxy = meshgrid(&[&v3, &y], MeshgridIndexing::Xy, false, true).unwrap();
    assert_dyn_matches_oracle(
        &mxy[0].clone().into(),
        oracle.arr("mesh_xy_x"),
        0.0,
        0.0,
        false,
        false,
        "mesh_xy_x",
    );
    assert_dyn_matches_oracle(
        &mxy[1].clone().into(),
        oracle.arr("mesh_xy_y"),
        0.0,
        0.0,
        false,
        false,
        "mesh_xy_y",
    );
    let mij = meshgrid(&[&v3, &y], MeshgridIndexing::Ij, true, false).unwrap();
    assert_dyn_matches_oracle(
        &mij[0].clone().into(),
        oracle.arr("mesh_ij_sp_x"),
        0.0,
        0.0,
        false,
        false,
        "mesh_ij_sp_x",
    );
    assert_dyn_matches_oracle(
        &mij[1].clone().into(),
        oracle.arr("mesh_ij_sp_y"),
        0.0,
        0.0,
        false,
        false,
        "mesh_ij_sp_y",
    );
    assert_dyn_matches_oracle(
        &indices(&[2, 3]).unwrap().into(),
        oracle.arr("indices"),
        0.0,
        0.0,
        false,
        false,
        "indices",
    );

    // Verify dynamic creation paths too
    assert_dyn_matches_oracle(
        &dyn_zeros(&[2, 3], DType::F64, Order::C).unwrap(),
        oracle.arr("zeros"),
        0.0,
        0.0,
        false,
        false,
        "dyn_zeros",
    );
    assert_dyn_matches_oracle(
        &dyn_ones(&[2, 3], DType::I32, Order::C).unwrap(),
        oracle.arr("ones"),
        0.0,
        0.0,
        false,
        false,
        "dyn_ones",
    );
    assert_dyn_matches_oracle(
        &dyn_full(&[2, 2], DynScalar::F64(-7.5), None, Order::C).unwrap(),
        oracle.arr("full"),
        0.0,
        0.0,
        false,
        false,
        "dyn_full",
    );
    assert_dyn_matches_oracle(
        &dyn_arange(
            DynScalar::F64(1.0),
            Some(DynScalar::F64(6.5)),
            Some(DynScalar::F64(1.5)),
            Some(DType::F64),
        )
        .unwrap(),
        oracle.arr("arange_pos"),
        1e-14,
        1e-14,
        false,
        false,
        "dyn_arange",
    );
    assert_dyn_matches_oracle(
        &dyn_linspace(
            DynScalar::F64(-2.0),
            DynScalar::F64(2.0),
            5,
            true,
            Some(DType::F64),
        )
        .unwrap(),
        oracle.arr("linspace_ep"),
        1e-14,
        1e-14,
        false,
        false,
        "dyn_linspace",
    );
    assert_dyn_matches_oracle(
        &dyn_logspace(
            DynScalar::F64(0.0),
            DynScalar::F64(3.0),
            4,
            true,
            2.0,
            Some(DType::F64),
        )
        .unwrap(),
        oracle.arr("logspace"),
        1e-13,
        1e-13,
        false,
        false,
        "dyn_logspace",
    );
    assert_dyn_matches_oracle(
        &dyn_geomspace(
            DynScalar::F64(1.0),
            DynScalar::F64(16.0),
            5,
            true,
            Some(DType::F64),
        )
        .unwrap(),
        oracle.arr("geomspace"),
        1e-13,
        1e-13,
        false,
        false,
        "dyn_geomspace",
    );
    assert_dyn_matches_oracle(
        &dyn_eye(3, Some(4), 1, DType::F64, Order::C).unwrap(),
        oracle.arr("eye"),
        0.0,
        0.0,
        false,
        false,
        "dyn_eye",
    );
    let dv3 = DynArray::from(v3.clone());
    let dy = DynArray::from(y.clone());
    let dm = dyn_meshgrid(&[&dv3, &dy], MeshgridIndexing::Xy, false, true).unwrap();
    assert_dyn_matches_oracle(
        &dm[0],
        oracle.arr("mesh_xy_x"),
        0.0,
        0.0,
        false,
        false,
        "dyn_mesh_x",
    );

    // Multi-dtype sanity including f16 and Complex64
    let c = zeros::<Complex64>(&[2, 2]).unwrap();
    let h = ones::<f16>(&[3]).unwrap();
    assert_aligned(&c, "c128_zeros");
    assert_aligned(&h, "f16_ones");
    assert_eq!(h.to_vec(), vec![f16::from_f32(1.0); 3]);

    // Huge allocation returns Err cleanly instead of panicking
    assert!(NdArray::<f64>::zeros(&[1usize << 61]).is_err());
    assert!(dyn_zeros(&[1usize << 61], DType::F64, Order::C).is_err());
}

#[test]
fn manipulation_views_join_split_pad_and_edit_against_numpy() {
    let base = arange(0i64, 12, 1).unwrap().reshape(&[3, 4]).unwrap();
    let base_f = arange(1.0f64, 13.0, 1.0).unwrap().reshape(&[3, 4]).unwrap();

    for (layout, view) in non_contiguous_2d_views(&base_f) {
        let [rows, cols] = [view.shape()[0], view.shape()[1]];
        let total = rows * cols;
        let py = format!(
            "{}
emit_arr('reshape_neg', a.reshape(-1))
emit_arr('ravel_c', np.ravel(a, order='C'))
emit_arr('ravel_f', np.ravel(a, order='F'))
emit_arr('flatten_f', a.flatten(order='F'))
emit_arr('transpose', a.T)
emit_arr('swapaxes', np.swapaxes(a, 0, 1))
emit_arr('expand', np.expand_dims(a, axis=(0, -1)))
emit_arr('squeeze', np.squeeze(np.expand_dims(a, axis=0), axis=0))
emit_arr('atleast3', np.atleast_3d(a))
emit_arr('concat_ax1', np.concatenate([a, a], axis=1))
emit_arr('stack_ax0', np.stack([a, a * 2], axis=0))
emit_arr('vstack', np.vstack([a, a]))
emit_arr('hstack', np.hstack([a, a]))
emit_arr('dstack', np.dstack([a, a]))
emit_arr('tile', np.tile(a, (2, 1)))
emit_arr('repeat_flat', np.repeat(a, 2))
emit_arr('flip_all', np.flip(a))
emit_arr('fliplr', np.fliplr(a))
emit_arr('flipud', np.flipud(a))
emit_arr('rot90', np.rot90(a, k=1, axes=(0, 1)))
emit_arr('roll', np.roll(a, shift=3, axis=1))
emit_arr('pad_const', np.pad(a, ((1, 2), (0, 1)), mode='constant', constant_values=-9.0))
emit_arr('pad_reflect', np.pad(a, 1, mode='reflect'))
emit_arr('pad_wrap', np.pad(a, 1, mode='wrap'))
emit_arr('resize', np.resize(a, (2, {})))
",
            py_assign_dyn("a", &DynArray::F64(view.clone())),
            total / 2 + 1
        );
        let oracle = run_oracle(&py);

        assert_dyn_matches_oracle(
            &reshape(&view, &[-1]).unwrap().into(),
            oracle.arr("reshape_neg"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/reshape"),
        );
        assert_dyn_matches_oracle(
            &ravel(&view).unwrap().into(),
            oracle.arr("ravel_c"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/ravel_c"),
        );
        assert_dyn_matches_oracle(
            &numpy_core::ravel_with_order(&view, Order::F).unwrap().into(),
            oracle.arr("ravel_f"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/ravel_f"),
        );
        assert_dyn_matches_oracle(
            &numpy_core::flatten_with_order(&view, Order::F).unwrap().into(),
            oracle.arr("flatten_f"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/flatten_f"),
        );
        assert_dyn_matches_oracle(
            &transpose(&view, None).unwrap().into(),
            oracle.arr("transpose"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/transpose"),
        );
        assert_dyn_matches_oracle(
            &swapaxes(&view, 0, 1).unwrap().into(),
            oracle.arr("swapaxes"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/swapaxes"),
        );
        let exp = expand_dims(&view, &[0, -1]).unwrap();
        assert_dyn_matches_oracle(
            &exp.clone().into(),
            oracle.arr("expand"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/expand"),
        );
        let exp0 = expand_dims(&view, &[0]).unwrap();
        assert_dyn_matches_oracle(
            &squeeze(&exp0, Some(&[0])).unwrap().into(),
            oracle.arr("squeeze"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/squeeze"),
        );
        assert_dyn_matches_oracle(
            &atleast_3d(&view).unwrap().into(),
            oracle.arr("atleast3"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/atleast3"),
        );
        assert_dyn_matches_oracle(
            &concatenate(&[&view, &view], Some(1)).unwrap().into(),
            oracle.arr("concat_ax1"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/concat"),
        );
        let doubled = view.clone();
        let doubled_arr = NdArray::from_shape_vec(
            doubled.shape(),
            doubled.to_vec().into_iter().map(|x| x * 2.0).collect(),
        )
        .unwrap();
        assert_dyn_matches_oracle(
            &stack(&[&view, &doubled_arr], 0).unwrap().into(),
            oracle.arr("stack_ax0"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/stack"),
        );
        assert_dyn_matches_oracle(
            &vstack(&[&view, &view]).unwrap().into(),
            oracle.arr("vstack"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/vstack"),
        );
        assert_dyn_matches_oracle(
            &hstack(&[&view, &view]).unwrap().into(),
            oracle.arr("hstack"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/hstack"),
        );
        assert_dyn_matches_oracle(
            &dstack(&[&view, &view]).unwrap().into(),
            oracle.arr("dstack"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/dstack"),
        );
        assert_dyn_matches_oracle(
            &tile(&view, &[2, 1]).unwrap().into(),
            oracle.arr("tile"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/tile"),
        );
        assert_dyn_matches_oracle(
            &repeat(&view, &[2], None).unwrap().into(),
            oracle.arr("repeat_flat"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/repeat"),
        );
        assert_dyn_matches_oracle(
            &flip(&view, None).unwrap().into(),
            oracle.arr("flip_all"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/flip"),
        );
        assert_dyn_matches_oracle(
            &fliplr(&view).unwrap().into(),
            oracle.arr("fliplr"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/fliplr"),
        );
        assert_dyn_matches_oracle(
            &flipud(&view).unwrap().into(),
            oracle.arr("flipud"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/flipud"),
        );
        assert_dyn_matches_oracle(
            &rot90(&view, 1, (0, 1)).unwrap().into(),
            oracle.arr("rot90"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/rot90"),
        );
        assert_dyn_matches_oracle(
            &roll(&view, 3, Some(1)).unwrap().into(),
            oracle.arr("roll"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/roll"),
        );
        assert_dyn_matches_oracle(
            &pad(
                &view,
                &PadWidth::Axes(vec![(1, 2), (0, 1)]),
                &PadMode::Constant(-9.0),
            )
            .unwrap()
            .into(),
            oracle.arr("pad_const"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/pad_const"),
        );
        assert_dyn_matches_oracle(
            &pad(&view, &PadWidth::Uniform(1), &PadMode::Reflect(ReflectType::Even))
                .unwrap()
                .into(),
            oracle.arr("pad_reflect"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/pad_reflect"),
        );
        assert_dyn_matches_oracle(
            &pad(&view, &PadWidth::Uniform(1), &PadMode::Wrap)
                .unwrap()
                .into(),
            oracle.arr("pad_wrap"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/pad_wrap"),
        );
        assert_dyn_matches_oracle(
            &resize(&view, &[2, total / 2 + 1]).unwrap().into(),
            oracle.arr("resize"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/resize"),
        );
    }

    // 3D moveaxis, rollaxis, split/dsplit/hsplit/vsplit/unstack, append/insert/delete/trim_zeros, column_stack
    let t3 = arange(0i64, 24, 1).unwrap().reshape(&[2, 3, 4]).unwrap();
    let v1 = NdArray::from_vec(vec![0i64, 0, 1, 2, 0, 3, 0, 0]);
    let b1 = NdArray::from_vec(vec![10i64, 20, 30]);
    let b2 = NdArray::from_vec(vec![1i64, 2, 3]);
    let s1 = atleast_1d(&NdArray::from_scalar(5i64)).unwrap();
    let py3 = format!(
        "{}\n{}\n{}\n{}\n{}\n\
emit_arr('moveaxis', np.moveaxis(t, [0, 2], [2, 1]))
emit_arr('rollaxis', np.rollaxis(t, 2, 0))
emit_arr('split_0', np.split(base, 2, axis=1)[0])
emit_arr('asplit_1', np.array_split(base, 3, axis=1)[1])
emit_arr('hsplit_0', np.hsplit(base, [1, 3])[1])
emit_arr('vsplit_1', np.vsplit(base, 3)[1])
emit_arr('dsplit_1', np.dsplit(t, 2)[1])
emit_arr('unstack_1', np.unstack(base, axis=0)[1])
emit_arr('colstack', np.column_stack([b1, b2]))
emit_arr('append', np.append(base, [[99, 98, 97, 96]], axis=0))
emit_arr('insert', np.insert(b1, [1, -1], 77))
emit_arr('delete', np.delete(base, 1, axis=1))
emit_arr('trim', np.trim_zeros(v1, 'fb'))
emit_arr('roll_axes', np.roll(t, (1, -2), axis=(0, 2)))
emit_arr('atleast2', np.atleast_2d(b1))
emit_arr('flatt', np.atleast_1d(base.flatten()))
",
        py_assign_dyn("t", &DynArray::I64(t3.clone())),
        py_assign_dyn("base", &DynArray::I64(base.clone())),
        py_assign_dyn("v1", &DynArray::I64(v1.clone())),
        py_assign_dyn("b1", &DynArray::I64(b1.clone())),
        py_assign_dyn("b2", &DynArray::I64(b2.clone()))
    );
    let o3 = run_oracle(&py3);

    assert_dyn_matches_oracle(
        &moveaxis(&t3, &[0, 2], &[2, 1]).unwrap().into(),
        o3.arr("moveaxis"),
        0.0,
        0.0,
        false,
        false,
        "moveaxis",
    );
    assert_dyn_matches_oracle(
        &rollaxis(&t3, 2, 0).unwrap().into(),
        o3.arr("rollaxis"),
        0.0,
        0.0,
        false,
        false,
        "rollaxis",
    );
    assert_dyn_matches_oracle(
        &split(&base, SplitSpec::Sections(2), 1).unwrap()[0].clone().into(),
        o3.arr("split_0"),
        0.0,
        0.0,
        false,
        false,
        "split",
    );
    assert_dyn_matches_oracle(
        &array_split(&base, SplitSpec::Sections(3), 1).unwrap()[1].clone().into(),
        o3.arr("asplit_1"),
        0.0,
        0.0,
        false,
        false,
        "array_split",
    );
    assert_dyn_matches_oracle(
        &hsplit(&base, SplitSpec::Indices(&[1, 3])).unwrap()[1].clone().into(),
        o3.arr("hsplit_0"),
        0.0,
        0.0,
        false,
        false,
        "hsplit",
    );
    assert_dyn_matches_oracle(
        &vsplit(&base, SplitSpec::Sections(3)).unwrap()[1].clone().into(),
        o3.arr("vsplit_1"),
        0.0,
        0.0,
        false,
        false,
        "vsplit",
    );
    assert_dyn_matches_oracle(
        &dsplit(&t3, SplitSpec::Sections(2)).unwrap()[1].clone().into(),
        o3.arr("dsplit_1"),
        0.0,
        0.0,
        false,
        false,
        "dsplit",
    );
    assert_dyn_matches_oracle(
        &unstack(&base, 0).unwrap()[1].clone().into(),
        o3.arr("unstack_1"),
        0.0,
        0.0,
        false,
        false,
        "unstack",
    );
    assert_dyn_matches_oracle(
        &column_stack(&[&b1, &b2]).unwrap().into(),
        o3.arr("colstack"),
        0.0,
        0.0,
        false,
        false,
        "column_stack",
    );
    let extra_row = NdArray::from_shape_vec(&[1, 4], vec![99i64, 98, 97, 96]).unwrap();
    assert_dyn_matches_oracle(
        &append(&base, &extra_row, Some(0)).unwrap().into(),
        o3.arr("append"),
        0.0,
        0.0,
        false,
        false,
        "append",
    );
    let val77 = NdArray::from_scalar(77i64);
    assert_dyn_matches_oracle(
        &insert(&b1, &InsertObj::Indices(&[1, -1]), &val77, None)
            .unwrap()
            .into(),
        o3.arr("insert"),
        0.0,
        0.0,
        false,
        false,
        "insert",
    );
    assert_dyn_matches_oracle(
        &delete(&base, &DeleteObj::Index(1), Some(1)).unwrap().into(),
        o3.arr("delete"),
        0.0,
        0.0,
        false,
        false,
        "delete",
    );
    assert_dyn_matches_oracle(
        &trim_zeros(&v1, TrimDirection::Both, None).unwrap().into(),
        o3.arr("trim"),
        0.0,
        0.0,
        false,
        false,
        "trim_zeros",
    );
    assert_dyn_matches_oracle(
        &roll_axes(&t3, &[1, -2], Some(&[0, 2])).unwrap().into(),
        o3.arr("roll_axes"),
        0.0,
        0.0,
        false,
        false,
        "roll_axes",
    );
    assert_dyn_matches_oracle(
        &atleast_2d(&b1).unwrap().into(),
        o3.arr("atleast2"),
        0.0,
        0.0,
        false,
        false,
        "atleast_2d",
    );
    let _ = s1;
    assert_dyn_matches_oracle(
        &flatten(&base).unwrap().into(),
        o3.arr("flatt"),
        0.0,
        0.0,
        false,
        false,
        "flatten",
    );
}

#[test]
fn indexing_and_advanced_subspace_against_numpy() {
    let base = arange(0i64, 12, 1).unwrap().reshape(&[3, 4]).unwrap();
    let cond = NdArray::from_shape_vec(
        &[3, 4],
        vec![
            true, false, true, false, false, true, false, true, true, true, false, false,
        ],
    )
    .unwrap();
    let other = full(&[3, 4], -100i64).unwrap();
    let idx = NdArray::from_vec(vec![0i64, -1, 5]);
    let idx_ax = NdArray::from_vec(vec![2i64, 0]);
    let arg_idx = NdArray::from_shape_vec(&[3, 2], vec![3i64, 1, 0, 2, 1, 3]).unwrap();

    for (layout, view) in non_contiguous_2d_views(&base) {
        let view_t_cond = NdArray::from_shape_vec(
            view.shape(),
            (0..view.len()).map(|i| i % 2 == 0 || i % 5 == 0).collect(),
        )
        .unwrap();
        let view_other = full(view.shape(), -50i64).unwrap();
        let py = format!(
            "{}\n{}\n{}\n\
emit_arr('take_wrap', np.take(a, [0, -1, 5], mode='wrap'))
emit_arr('take_clip', np.take(a, [0, -1, 5], mode='clip'))
emit_arr('take_ax0', np.take(a, [-1, 0], axis=0, mode='raise'))
emit_arr('diag', np.diagonal(a, offset=1, axis1=0, axis2=1))
emit_arr('nonzero_0', np.nonzero(c)[0])
emit_arr('nonzero_1', np.nonzero(c)[1])
emit_arr('argwhere', np.argwhere(c))
emit_arr('flatnonzero', np.flatnonzero(c))
emit_arr('where', np.where(c, a, o))
emit_arr('select', np.select([c], [a], default=-7))
emit_arr('extract', np.extract(c, a))
emit_arr('bool_mask', a[c])
",
            py_assign_dyn("a", &DynArray::I64(view.clone())),
            py_assign_dyn("c", &DynArray::Bool(view_t_cond.clone())),
            py_assign_dyn("o", &DynArray::I64(view_other.clone())),
        );
        let oracle = run_oracle(&py);

        assert_dyn_matches_oracle(
            &take(&view, &idx, None, IndexMode::Wrap).unwrap().into(),
            oracle.arr("take_wrap"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/take_wrap"),
        );
        assert_dyn_matches_oracle(
            &take(&view, &idx, None, IndexMode::Clip).unwrap().into(),
            oracle.arr("take_clip"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/take_clip"),
        );
        let ax0_idx = NdArray::from_vec(vec![-1i64, 0]);
        assert_dyn_matches_oracle(
            &take(&view, &ax0_idx, Some(0), IndexMode::Raise).unwrap().into(),
            oracle.arr("take_ax0"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/take_ax0"),
        );
        assert_dyn_matches_oracle(
            &diagonal(&view, 1, 0, 1).unwrap().into(),
            oracle.arr("diag"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/diagonal"),
        );
        let nz = nonzero(&view_t_cond).unwrap();
        assert_dyn_matches_oracle(
            &nz[0].clone().into(),
            oracle.arr("nonzero_0"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/nonzero0"),
        );
        assert_dyn_matches_oracle(
            &nz[1].clone().into(),
            oracle.arr("nonzero_1"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/nonzero1"),
        );
        assert_dyn_matches_oracle(
            &argwhere(&view_t_cond).unwrap().into(),
            oracle.arr("argwhere"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/argwhere"),
        );
        assert_dyn_matches_oracle(
            &flatnonzero(&view_t_cond).unwrap().into(),
            oracle.arr("flatnonzero"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/flatnonzero"),
        );
        assert_dyn_matches_oracle(
            &where_select(&view_t_cond, &view, &view_other).unwrap().into(),
            oracle.arr("where"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/where"),
        );
        assert_dyn_matches_oracle(
            &select(&[&view_t_cond], &[&view], -7).unwrap().into(),
            oracle.arr("select"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/select"),
        );
        assert_dyn_matches_oracle(
            &extract(&view_t_cond, &view).unwrap().into(),
            oracle.arr("extract"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/extract"),
        );
        assert_dyn_matches_oracle(
            &boolean_mask(&view, &view_t_cond).unwrap().into(),
            oracle.arr("bool_mask"),
            0.0,
            0.0,
            false,
            false,
            &format!("{layout}/bool_mask"),
        );
    }

    // Mutating / multi-index / subspace routines
    let r_idx = NdArray::from_vec(vec![0i64, 2]);
    let c_idx = NdArray::from_vec(vec![3i64, 1]);
    let sel = NdArray::from_shape_vec(&[2, 2], vec![0i64, 1, 1, 0]).unwrap();
    let ch0 = full(&[2, 2], 10i64).unwrap();
    let ch1 = full(&[2, 2], 20i64).unwrap();
    let comp_cond = NdArray::from_vec(vec![true, false, true]);
    let mut put_arr = base.clone();
    put(
        &mut put_arr,
        &NdArray::from_vec(vec![1i64, -2]),
        &NdArray::from_vec(vec![88i64, 99]),
        IndexMode::Raise,
    )
    .unwrap();
    let mut put_ax_arr = base.clone();
    put_along_axis(
        &mut put_ax_arr,
        &arg_idx,
        &NdArray::from_scalar(-5i64),
        Some(1),
    )
    .unwrap();
    let mut putm_arr = base.clone();
    putmask(&mut putm_arr, &cond, &other).unwrap();
    let mut fd_arr = base.clone();
    fill_diagonal(&mut fd_arr, &NdArray::from_scalar(42i64), false).unwrap();

    let adv_row_idx = NdArray::from_vec(vec![2i64, 0]);
    let adv_col_mask = NdArray::from_vec(vec![true, false, true, false]);
    let t3 = arange(0i64, 24, 1).unwrap().reshape(&[2, 3, 4]).unwrap();
    let bmask_row = NdArray::from_vec(vec![false, true]);
    let bmask_ax2 = NdArray::from_vec(vec![true, false, false, true]);

    let py2 = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\
emit_arr('take_along', np.take_along_axis(base, arg_idx, axis=1))
a1 = base.copy(); np.put(a1, [1, -2], [88, 99]); emit_arr('put', a1)
a2 = base.copy(); np.put_along_axis(a2, arg_idx, -5, axis=1); emit_arr('put_along', a2)
a3 = base.copy(); np.putmask(a3, cond, other); emit_arr('putmask', a3)
a4 = base.copy(); np.fill_diagonal(a4, 42); emit_arr('fill_diag', a4)
emit_arr('compress', np.compress([True, False, True], base, axis=0))
emit_arr('choose', np.choose(sel, [ch0, ch1]))
unr = np.unravel_index([2, 7, 11], (3, 4), order='F')
emit_arr('unravel0', unr[0])
emit_arr('unravel1', unr[1])
emit_arr('ravel_idx', np.ravel_multi_index(([0, 2], [3, 1]), (3, 4), order='F'))
emit_arr('fancy', base[[0, 2], [3, 1]])
emit_arr('adv_split', t3[np.ix_([False, True], [0, 2]) if False else (np.array([False, True]), slice(None), np.array([True, False, False, True]))])
emit_arr('adv_mix', base[:, None, [True, False, True, False]])
",
        py_assign_dyn("base", &DynArray::I64(base.clone())),
        py_assign_dyn("arg_idx", &DynArray::I64(arg_idx.clone())),
        py_assign_dyn("cond", &DynArray::Bool(cond.clone())),
        py_assign_dyn("other", &DynArray::I64(other.clone())),
        py_assign_dyn("sel", &DynArray::I64(sel.clone())),
        py_assign_dyn("ch0", &DynArray::I64(ch0.clone())),
        py_assign_dyn("ch1", &DynArray::I64(ch1.clone())),
        py_assign_dyn("t3", &DynArray::I64(t3.clone())),
    );
    let o2 = run_oracle(&py2);

    assert_dyn_matches_oracle(
        &take_along_axis(&base, &arg_idx, Some(1)).unwrap().into(),
        o2.arr("take_along"),
        0.0,
        0.0,
        false,
        false,
        "take_along_axis",
    );
    assert_dyn_matches_oracle(&put_arr.into(), o2.arr("put"), 0.0, 0.0, false, false, "put");
    assert_dyn_matches_oracle(
        &put_ax_arr.into(),
        o2.arr("put_along"),
        0.0,
        0.0,
        false,
        false,
        "put_along",
    );
    assert_dyn_matches_oracle(
        &putm_arr.into(),
        o2.arr("putmask"),
        0.0,
        0.0,
        false,
        false,
        "putmask",
    );
    assert_dyn_matches_oracle(
        &fd_arr.into(),
        o2.arr("fill_diag"),
        0.0,
        0.0,
        false,
        false,
        "fill_diagonal",
    );
    assert_dyn_matches_oracle(
        &compress(&comp_cond, &base, Some(0)).unwrap().into(),
        o2.arr("compress"),
        0.0,
        0.0,
        false,
        false,
        "compress",
    );
    assert_dyn_matches_oracle(
        &choose(&sel, &[&ch0, &ch1], IndexMode::Raise).unwrap().into(),
        o2.arr("choose"),
        0.0,
        0.0,
        false,
        false,
        "choose",
    );
    let unr = unravel_index(&NdArray::from_vec(vec![2i64, 7, 11]), &[3, 4], Order::F).unwrap();
    assert_dyn_matches_oracle(
        &unr[0].clone().into(),
        o2.arr("unravel0"),
        0.0,
        0.0,
        false,
        false,
        "unravel0",
    );
    assert_dyn_matches_oracle(
        &unr[1].clone().into(),
        o2.arr("unravel1"),
        0.0,
        0.0,
        false,
        false,
        "unravel1",
    );
    assert_dyn_matches_oracle(
        &ravel_multi_index(&[&r_idx, &c_idx], &[3, 4], IndexMode::Raise, Order::F)
            .unwrap()
            .into(),
        o2.arr("ravel_idx"),
        0.0,
        0.0,
        false,
        false,
        "ravel_multi_index",
    );
    assert_dyn_matches_oracle(
        &index_fancy(&base, &[&r_idx, &c_idx]).unwrap().into(),
        o2.arr("fancy"),
        0.0,
        0.0,
        false,
        false,
        "index_fancy",
    );
    assert_dyn_matches_oracle(
        &index_advanced(
            &t3,
            &[
                FancyArg::BoolMask(&bmask_row),
                FancyArg::Slice(Slice::full()),
                FancyArg::BoolMask(&bmask_ax2),
            ],
        )
        .unwrap()
        .into(),
        o2.arr("adv_split"),
        0.0,
        0.0,
        false,
        false,
        "index_advanced_split",
    );
    assert_dyn_matches_oracle(
        &index_advanced(
            &base,
            &[
                FancyArg::Slice(Slice::full()),
                FancyArg::NewAxis,
                FancyArg::BoolMask(&adv_col_mask),
            ],
        )
        .unwrap()
        .into(),
        o2.arr("adv_mix"),
        0.0,
        0.0,
        false,
        false,
        "index_advanced_mix",
    );
    let _ = (idx_ax, adv_row_idx);
}

#[test]
fn memory_sharing_copyto_and_einsum_against_numpy() {
    let mut dst = zeros::<f64>(&[3, 4]).unwrap();
    let src = arange(1.0f64, 13.0, 1.0).unwrap().reshape(&[3, 4]).unwrap();
    let mask = NdArray::from_vec(vec![true, false, true, false]);
    copyto(&mut dst, &src, Some(&mask)).unwrap();

    // Verify NdArray::copy isolation and memory sharing rules
    let s1 = src
        .slice(&[SliceArg::Slice(Slice::new(None, None, 2)), SliceArg::Slice(Slice::full())])
        .unwrap();
    let s2 = src
        .slice(&[
            SliceArg::Slice(Slice::new(Some(1), None, 2)),
            SliceArg::Slice(Slice::full()),
        ])
        .unwrap();
    let copied = src.copy(Order::F);
    assert!(may_share_memory(&src, &s1));
    assert!(shares_memory(&src, &s1));
    assert!(!shares_memory(&s1, &s2));
    assert!(!may_share_memory(&src, &copied));

    // Einsum: trace, diagonal, transpose, outer, matmul, batched, chain contraction, ellipsis, complex
    let a = arange(1.0f64, 13.0, 1.0).unwrap().reshape(&[3, 4]).unwrap();
    let b = arange(1.0f64, 21.0, 1.0).unwrap().reshape(&[4, 5]).unwrap();
    let c = arange(0.5f64, 5.5, 0.5).unwrap().reshape(&[5, 2]).unwrap();
    let sq = arange(1.0f64, 10.0, 1.0).unwrap().reshape(&[3, 3]).unwrap();
    let bat_a = arange(1.0f64, 25.0, 1.0).unwrap().reshape(&[2, 3, 4]).unwrap();
    let bat_b = arange(1.0f64, 41.0, 1.0).unwrap().reshape(&[2, 4, 5]).unwrap();

    let py = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\
d = np.zeros((3, 4), dtype=np.float64)
np.copyto(d, src, where=mask)
emit_arr('copyto', d)
emit_arr('trace', np.einsum('ii->', sq))
emit_arr('diag', np.einsum('ii->i', sq))
emit_arr('outer', np.einsum('i,j->ij', src[0], src[1]))
emit_arr('matmul_t', np.einsum('ij,jk->ik', a.T.T, b))
emit_arr('chain', np.einsum('ij,jk,kl->il', a, b, c, optimize='optimal'))
emit_arr('ellipsis', np.einsum('...ij,...jk->...ik', bat_a, bat_b))
",
        py_assign_dyn("src", &DynArray::F64(src.clone())),
        py_assign_dyn("mask", &DynArray::Bool(mask.clone())),
        py_assign_dyn("sq", &DynArray::F64(sq.clone())),
        py_assign_dyn("a", &DynArray::F64(a.clone())),
        py_assign_dyn("b", &DynArray::F64(b.clone())),
        py_assign_dyn("c", &DynArray::F64(c.clone())),
        py_assign_dyn("bat_a", &DynArray::F64(bat_a.clone())),
        py_assign_dyn("bat_b", &DynArray::F64(bat_b.clone())),
    );
    let oracle = run_oracle(&py);

    assert_dyn_matches_oracle(&dst.into(), oracle.arr("copyto"), 0.0, 0.0, false, false, "copyto");
    assert_dyn_matches_oracle(
        &einsum("ii->", &[&sq]).unwrap().into(),
        oracle.arr("trace"),
        1e-13,
        1e-13,
        false,
        false,
        "einsum_trace",
    );
    assert_dyn_matches_oracle(
        &einsum("ii->i", &[&sq]).unwrap().into(),
        oracle.arr("diag"),
        1e-13,
        1e-13,
        false,
        false,
        "einsum_diag",
    );
    let r0 = src.slice(&[SliceArg::Index(0), SliceArg::Slice(Slice::full())]).unwrap();
    let r1 = src.slice(&[SliceArg::Index(1), SliceArg::Slice(Slice::full())]).unwrap();
    assert_dyn_matches_oracle(
        &einsum("i,j->ij", &[&r0, &r1]).unwrap().into(),
        oracle.arr("outer"),
        1e-13,
        1e-13,
        false,
        false,
        "einsum_outer",
    );
    let a_view = a.t().t();
    assert_dyn_matches_oracle(
        &einsum("ij,jk->ik", &[&a_view, &b]).unwrap().into(),
        oracle.arr("matmul_t"),
        1e-12,
        1e-12,
        false,
        false,
        "einsum_matmul",
    );
    for opt in [
        EinsumOptimize::None,
        EinsumOptimize::Greedy,
        EinsumOptimize::Optimal,
        EinsumOptimize::Auto,
    ] {
        assert_dyn_matches_oracle(
            &einsum_with_optimize("ij,jk,kl->il", &[&a, &b, &c], opt)
                .unwrap()
                .into(),
            oracle.arr("chain"),
            1e-11,
            1e-11,
            false,
            false,
            "einsum_chain",
        );
    }
    let da = DynArray::from(bat_a);
    let db = DynArray::from(bat_b);
    assert_dyn_matches_oracle(
        &dyn_einsum("...ij,...jk->...ik", &[&da, &db], EinsumOptimize::Greedy).unwrap(),
        oracle.arr("ellipsis"),
        1e-11,
        1e-11,
        false,
        false,
        "dyn_einsum_ellipsis",
    );
}

#[test]
fn round2_defects_parity_against_numpy() {
    let py = r#"
a = np.array([[1, 2, 3]], dtype=np.int32)
s = a[::-(2**63), :]
emit_arr("slice_min", s)
emit_arr("flip_min", np.flip(s, axis=0))

v = np.array([10, 20, 30], dtype=np.int32)
vs = v[::-(2**63)]
emit_arr("v_slice_min", vs)
emit_arr("v_flip_min", np.flip(vs, axis=0))

x = np.array([10, 20, 30], dtype=np.int64)
emit_arr("ins_max", np.insert(x, slice(1, 2, 2**63 - 1), 99))
emit_arr("del_max", np.delete(x, slice(1, 2, 2**63 - 1)))

emit_arr("clog_base0", np.logspace(-1+0j, 1+0j, 3, endpoint=True, base=0.0, dtype=np.complex128))
emit_arr("clog_base0_0p1j_to_1p1j", np.logspace(0+1j, 1+1j, 2, endpoint=True, base=0.0, dtype=np.complex128))
emit_arr("clog_base0_half_m3j", np.logspace(0.5-3j, 0.5-3j, 1, endpoint=True, base=0.0, dtype=np.complex128))
emit_arr("clog_base0_1p2j", np.logspace(1+2j, 1+2j, 1, endpoint=True, base=0.0, dtype=np.complex128))
emit_arr("clog_base0_m1p2j", np.logspace(-1+2j, -1+2j, 1, endpoint=True, base=0.0, dtype=np.complex128))
emit_arr("clog_base0_0j", np.logspace(0+0j, 0+0j, 1, endpoint=True, base=0.0, dtype=np.complex128))

try:
    np.fill_diagonal(np.zeros((2, 3, 4), dtype=np.int64), np.array([], dtype=np.int64))
except Exception as e:
    emit_err("fd_3d_empty", e)
"#;
    let o = run_oracle(py);

    let a = NdArray::from_shape_vec(&[1, 3], vec![1i32, 2, 3]).unwrap();
    let s = a
        .slice(&[
            SliceArg::Slice(Slice::new(None, None, isize::MIN)),
            SliceArg::Slice(Slice::full()),
        ])
        .unwrap();
    let f = s.flip(Some(&[0])).unwrap();
    assert_dyn_matches_oracle(&s.into(), o.arr("slice_min"), 0.0, 0.0, false, true, "slice_min");
    assert_dyn_matches_oracle(&f.into(), o.arr("flip_min"), 0.0, 0.0, false, true, "flip_min");

    let v = NdArray::from_vec(vec![10i32, 20, 30]);
    let vs = v
        .slice(&[SliceArg::Slice(Slice::new(None, None, isize::MIN))])
        .unwrap();
    let vf = vs.flip(Some(&[0])).unwrap();
    assert_dyn_matches_oracle(&vs.into(), o.arr("v_slice_min"), 0.0, 0.0, false, true, "v_slice_min");
    assert_dyn_matches_oracle(&vf.into(), o.arr("v_flip_min"), 0.0, 0.0, false, true, "v_flip_min");

    let x = NdArray::from_vec(vec![10i64, 20, 30]);
    let val = NdArray::from_scalar(99i64);
    let ins = insert(&x, &InsertObj::Slice(Slice::new(Some(1), Some(2), isize::MAX)), &val, None).unwrap();
    let del = delete(&x, &DeleteObj::Slice(Slice::new(Some(1), Some(2), isize::MAX)), None).unwrap();
    assert_dyn_matches_oracle(&ins.into(), o.arr("ins_max"), 0.0, 0.0, false, true, "ins_max");
    assert_dyn_matches_oracle(&del.into(), o.arr("del_max"), 0.0, 0.0, false, true, "del_max");

    let clog = logspace(
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        3,
        true,
        0.0,
    )
    .unwrap();
    assert_dyn_matches_oracle(&clog.into(), o.arr("clog_base0"), 0.0, 0.0, true, true, "clog_base0");

    let clog_0p1j_to_1p1j = logspace(
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 1.0),
        2,
        true,
        0.0,
    )
    .unwrap();
    assert_dyn_matches_oracle(
        &clog_0p1j_to_1p1j.into(),
        o.arr("clog_base0_0p1j_to_1p1j"),
        0.0,
        0.0,
        true,
        true,
        "clog_base0_0p1j_to_1p1j",
    );
    let clog_half_m3j = logspace(
        Complex64::new(0.5, -3.0),
        Complex64::new(0.5, -3.0),
        1,
        true,
        0.0,
    )
    .unwrap();
    assert_dyn_matches_oracle(
        &clog_half_m3j.into(),
        o.arr("clog_base0_half_m3j"),
        0.0,
        0.0,
        true,
        true,
        "clog_base0_half_m3j",
    );
    let clog_1p2j = logspace(
        Complex64::new(1.0, 2.0),
        Complex64::new(1.0, 2.0),
        1,
        true,
        0.0,
    )
    .unwrap();
    assert_dyn_matches_oracle(
        &clog_1p2j.into(),
        o.arr("clog_base0_1p2j"),
        0.0,
        0.0,
        true,
        true,
        "clog_base0_1p2j",
    );
    let clog_m1p2j = logspace(
        Complex64::new(-1.0, 2.0),
        Complex64::new(-1.0, 2.0),
        1,
        true,
        0.0,
    )
    .unwrap();
    assert_dyn_matches_oracle(
        &clog_m1p2j.into(),
        o.arr("clog_base0_m1p2j"),
        0.0,
        0.0,
        true,
        true,
        "clog_base0_m1p2j",
    );
    let clog_0j = logspace(
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        1,
        true,
        0.0,
    )
    .unwrap();
    assert_dyn_matches_oracle(
        &clog_0j.into(),
        o.arr("clog_base0_0j"),
        0.0,
        0.0,
        true,
        true,
        "clog_base0_0j",
    );

    assert!(o.errors.contains_key("fd_3d_empty"));
    let mut a3 = NdArray::<i64>::zeros(&[2, 3, 4]).unwrap();
    let empty_val = NdArray::<i64>::zeros(&[0]).unwrap();
    assert!(matches!(
        fill_diagonal(&mut a3, &empty_val, false),
        Err(numpy_types::NumPyError::ShapeMismatch(_))
    ));
}
