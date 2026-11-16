//! Dynamic (`DynArray`) namespace mirroring top-level NumPy module calls (`numpy::dynamic::zeros`,
//! `numpy::dynamic::concatenate`, `numpy::dynamic::einsum`, etc.).

pub use crate::creation::{
    dyn_arange as arange, dyn_diag as diag, dyn_diagflat as diagflat, dyn_empty as empty,
    dyn_empty_like as empty_like, dyn_eye as eye, dyn_from_fn as from_fn,
    dyn_from_fn_par as from_fn_par, dyn_full as full, dyn_full_like as full_like,
    dyn_geomspace as geomspace, dyn_identity as identity, dyn_indices as indices,
    dyn_indices_sparse as indices_sparse, dyn_linspace as linspace,
    dyn_linspace_retstep as linspace_retstep, dyn_logspace as logspace,
    dyn_meshgrid as meshgrid, dyn_ones as ones, dyn_ones_like as ones_like, dyn_tri as tri,
    dyn_tril as tril, dyn_triu as triu, dyn_vander as vander, dyn_zeros as zeros,
    dyn_zeros_like as zeros_like, MeshgridIndexing,
};
pub use crate::einsum::{
    dyn_einsum as einsum, einsum_path, einsum_path_shapes, EinsumContraction, EinsumOptimize,
    EinsumPathInfo,
};
pub use crate::format::{
    dyn_array2string as array2string, dyn_array2string_default as array2string_default,
    dyn_array_repr as array_repr, dyn_array_str as array_str, FloatMode, PrintOptions, SignMode,
};
pub use crate::indexing::{
    dyn_argwhere as argwhere, dyn_boolean_mask as boolean_mask,
    dyn_boolean_mask_set as boolean_mask_set, dyn_choose as choose, dyn_compress as compress,
    dyn_diagonal as diagonal, dyn_extract as extract, dyn_fill_diagonal as fill_diagonal,
    dyn_flatnonzero as flatnonzero, dyn_index_advanced as index_advanced,
    dyn_index_fancy as index_fancy, dyn_index_fancy_set as index_fancy_set,
    dyn_nonzero as nonzero, dyn_put as put, dyn_put_along_axis as put_along_axis,
    dyn_putmask as putmask, dyn_ravel_multi_index as ravel_multi_index, dyn_select as select,
    dyn_take as take, dyn_take_along_axis as take_along_axis,
    dyn_unravel_index as unravel_index, dyn_where_select as where_select, DynFancyArg,
    IndexMode,
};
pub use crate::manipulation::{
    dyn_append as append, dyn_array_split as array_split, dyn_atleast_1d as atleast_1d,
    dyn_atleast_2d as atleast_2d, dyn_atleast_3d as atleast_3d, dyn_block as block,
    dyn_column_stack as column_stack, dyn_concat as concat, dyn_concatenate as concatenate,
    dyn_delete as delete, dyn_dsplit as dsplit, dyn_dstack as dstack,
    dyn_expand_dims as expand_dims, dyn_flatten as flatten,
    dyn_flatten_with_order as flatten_with_order, dyn_flip as flip, dyn_fliplr as fliplr,
    dyn_flipud as flipud, dyn_hsplit as hsplit, dyn_hstack as hstack, dyn_insert as insert,
    dyn_moveaxis as moveaxis, dyn_pad as pad, dyn_ravel as ravel,
    dyn_ravel_with_order as ravel_with_order, dyn_repeat as repeat, dyn_reshape as reshape,
    dyn_reshape_with_order as reshape_with_order, dyn_resize as resize, dyn_roll as roll,
    dyn_roll_axes as roll_axes, dyn_rollaxis as rollaxis, dyn_rot90 as rot90, dyn_split as split,
    dyn_squeeze as squeeze, dyn_stack as stack, dyn_swapaxes as swapaxes, dyn_tile as tile,
    dyn_transpose as transpose, dyn_trim_zeros as trim_zeros, dyn_unstack as unstack,
    dyn_vsplit as vsplit, dyn_vstack as vstack, DynBlockNode, DynDeleteObj, DynInsertObj,
    DynPadMode, PadWidth, ReflectType, SplitSpec, StatLength, TrimDirection,
};
pub use crate::memory::{
    dyn_copyto as copyto, may_share_memory, memmap, shares_memory, DynMemmap, MemmapMode,
    MemoryInspect, MemorySpan,
};
pub use crate::util::DynScalar;
