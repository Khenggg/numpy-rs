//! Method-chaining extension traits `NdArrayExt<T>` and `DynArrayExt` for `NdArray<T>` and `DynArray`.

use crate::creation::{diag, diagflat, tril, triu, vander, CreationElement};
use crate::format::{
    array2string, array2string_default, array_repr, array_str, dyn_array2string,
    dyn_array2string_default, dyn_array_repr, dyn_array_str, FormatElement, PrintOptions,
};
use crate::indexing::{
    argwhere, boolean_mask, boolean_mask_set, compress, diagonal, dyn_argwhere,
    dyn_boolean_mask, dyn_boolean_mask_set, dyn_compress, dyn_diagonal, dyn_fill_diagonal,
    dyn_flatnonzero, dyn_index_advanced, dyn_index_fancy, dyn_index_fancy_set, dyn_nonzero,
    dyn_put, dyn_put_along_axis, dyn_putmask, dyn_take, dyn_take_along_axis, fill_diagonal,
    flatnonzero, index_advanced, index_fancy, index_fancy_set, nonzero, put, put_along_axis,
    putmask, take, take_along_axis, DynFancyArg, FancyArg, IndexMode,
};
use crate::manipulation::{
    append, array_split, atleast_1d, atleast_2d, atleast_3d, delete, dsplit, dyn_append,
    dyn_array_split, dyn_atleast_1d, dyn_atleast_2d, dyn_atleast_3d, dyn_delete, dyn_dsplit,
    dyn_expand_dims, dyn_flatten, dyn_flatten_with_order, dyn_flip, dyn_fliplr, dyn_flipud,
    dyn_hsplit, dyn_insert, dyn_moveaxis, dyn_pad, dyn_ravel, dyn_ravel_with_order, dyn_repeat,
    dyn_reshape_with_order, dyn_resize, dyn_roll, dyn_roll_axes, dyn_rollaxis, dyn_rot90,
    dyn_split, dyn_squeeze, dyn_swapaxes, dyn_tile, dyn_trim_zeros, dyn_unstack, dyn_vsplit,
    expand_dims, flatten, flatten_with_order, flip, fliplr, flipud, hsplit, insert, moveaxis,
    pad, ravel_with_order, repeat, reshape_with_order, resize, roll, roll_axes, rollaxis, rot90,
    split, squeeze, tile, trim_zeros, unstack, vsplit, DeleteObj, DynDeleteObj, DynInsertObj,
    DynPadMode, InsertObj, PadElement, PadMode, PadWidth, SplitSpec, TrimDirection,
};
use crate::memory::{copyto, dyn_copyto, may_share_memory, shares_memory, MemoryInspect};
use numpy_types::{DynArray, NdArray, Order, Result};

/// Extension methods provided by `numpy-core` on typed `NdArray<T>`.
pub trait NdArrayExt<T: CreationElement + PadElement + FormatElement> {
    fn reshape_with_order(&self, newshape: &[isize], order: Order) -> Result<NdArray<T>>;
    fn ravel_with_order(&self, order: Order) -> Result<NdArray<T>>;
    fn flatten(&self) -> Result<NdArray<T>>;
    fn flatten_with_order(&self, order: Order) -> Result<NdArray<T>>;
    fn moveaxis(&self, source: &[isize], destination: &[isize]) -> Result<NdArray<T>>;
    fn rollaxis(&self, axis: isize, start: isize) -> Result<NdArray<T>>;
    fn squeeze_axes(&self, axis: Option<&[isize]>) -> Result<NdArray<T>>;
    fn expand_dims_axes(&self, axes: &[isize]) -> Result<NdArray<T>>;
    fn atleast_1d(&self) -> Result<NdArray<T>>;
    fn atleast_2d(&self) -> Result<NdArray<T>>;
    fn atleast_3d(&self) -> Result<NdArray<T>>;
    fn split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<NdArray<T>>>;
    fn array_split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<NdArray<T>>>;
    fn hsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>>;
    fn vsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>>;
    fn dsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>>;
    fn unstack(&self, axis: isize) -> Result<Vec<NdArray<T>>>;
    fn tile(&self, reps: &[usize]) -> Result<NdArray<T>>;
    fn repeat(&self, repeats: &[usize], axis: Option<isize>) -> Result<NdArray<T>>;
    fn flip_axes(&self, axis: Option<&[isize]>) -> Result<NdArray<T>>;
    fn fliplr(&self) -> Result<NdArray<T>>;
    fn flipud(&self) -> Result<NdArray<T>>;
    fn rot90(&self, k: isize, axes: (isize, isize)) -> Result<NdArray<T>>;
    fn roll(&self, shift: isize, axis: Option<isize>) -> Result<NdArray<T>>;
    fn roll_axes(&self, shifts: &[isize], axes: Option<&[isize]>) -> Result<NdArray<T>>;
    fn pad(&self, pad_width: &PadWidth, mode: &PadMode<T>) -> Result<NdArray<T>>;
    fn resize(&self, new_shape: &[usize]) -> Result<NdArray<T>>;
    fn append(&self, values: &NdArray<T>, axis: Option<isize>) -> Result<NdArray<T>>;
    fn insert(&self, obj: &InsertObj<'_>, values: &NdArray<T>, axis: Option<isize>) -> Result<NdArray<T>>;
    fn delete(&self, obj: &DeleteObj<'_>, axis: Option<isize>) -> Result<NdArray<T>>;
    fn trim_zeros(&self, trim: TrimDirection, axis: Option<&[isize]>) -> Result<NdArray<T>>;
    fn diag(&self, k: isize) -> Result<NdArray<T>>;
    fn diagflat(&self, k: isize) -> Result<NdArray<T>>;
    fn tril(&self, k: isize) -> Result<NdArray<T>>;
    fn triu(&self, k: isize) -> Result<NdArray<T>>;
    fn vander(&self, n: Option<usize>, increasing: bool) -> Result<NdArray<T>>;
    fn take(&self, indices: &NdArray<i64>, axis: Option<isize>, mode: IndexMode) -> Result<NdArray<T>>;
    fn take_along_axis(&self, indices: &NdArray<i64>, axis: Option<isize>) -> Result<NdArray<T>>;
    fn put(&mut self, ind: &NdArray<i64>, v: &NdArray<T>, mode: IndexMode) -> Result<()>;
    fn put_along_axis(&mut self, indices: &NdArray<i64>, values: &NdArray<T>, axis: Option<isize>) -> Result<()>;
    fn putmask(&mut self, mask: &NdArray<bool>, values: &NdArray<T>) -> Result<()>;
    fn compress(&self, condition: &NdArray<bool>, axis: Option<isize>) -> Result<NdArray<T>>;
    fn nonzero(&self) -> Result<Vec<NdArray<i64>>>;
    fn argwhere(&self) -> Result<NdArray<i64>>;
    fn flatnonzero(&self) -> Result<NdArray<i64>>;
    fn diagonal(&self, offset: isize, axis1: isize, axis2: isize) -> Result<NdArray<T>>;
    fn fill_diagonal(&mut self, val: &NdArray<T>, wrap: bool) -> Result<()>;
    fn boolean_mask(&self, mask: &NdArray<bool>) -> Result<NdArray<T>>;
    fn boolean_mask_set(&mut self, mask: &NdArray<bool>, values: &NdArray<T>) -> Result<()>;
    fn index_fancy(&self, indices: &[&NdArray<i64>]) -> Result<NdArray<T>>;
    fn index_fancy_set(&mut self, indices: &[&NdArray<i64>], values: &NdArray<T>) -> Result<()>;
    fn index_advanced(&self, args: &[FancyArg<'_>]) -> Result<NdArray<T>>;
    fn copy_from(&mut self, src: &NdArray<T>, where_mask: Option<&NdArray<bool>>) -> Result<()>;
    fn shares_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool;
    fn may_share_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool;
    fn array2string(&self, options: &PrintOptions) -> Result<String>;
    fn array2string_default(&self) -> Result<String>;
    fn array_str(&self, options: &PrintOptions) -> Result<String>;
    fn array_repr(&self, options: &PrintOptions) -> Result<String>;
}

impl<T: CreationElement + PadElement + FormatElement> NdArrayExt<T> for NdArray<T> {
    fn reshape_with_order(&self, newshape: &[isize], order: Order) -> Result<NdArray<T>> {
        reshape_with_order(self, newshape, order)
    }
    fn ravel_with_order(&self, order: Order) -> Result<NdArray<T>> {
        ravel_with_order(self, order)
    }
    fn flatten(&self) -> Result<NdArray<T>> {
        flatten(self)
    }
    fn flatten_with_order(&self, order: Order) -> Result<NdArray<T>> {
        flatten_with_order(self, order)
    }
    fn moveaxis(&self, source: &[isize], destination: &[isize]) -> Result<NdArray<T>> {
        moveaxis(self, source, destination)
    }
    fn rollaxis(&self, axis: isize, start: isize) -> Result<NdArray<T>> {
        rollaxis(self, axis, start)
    }
    fn squeeze_axes(&self, axis: Option<&[isize]>) -> Result<NdArray<T>> {
        squeeze(self, axis)
    }
    fn expand_dims_axes(&self, axes: &[isize]) -> Result<NdArray<T>> {
        expand_dims(self, axes)
    }
    fn atleast_1d(&self) -> Result<NdArray<T>> {
        atleast_1d(self)
    }
    fn atleast_2d(&self) -> Result<NdArray<T>> {
        atleast_2d(self)
    }
    fn atleast_3d(&self) -> Result<NdArray<T>> {
        atleast_3d(self)
    }
    fn split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<NdArray<T>>> {
        split(self, spec, axis)
    }
    fn array_split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<NdArray<T>>> {
        array_split(self, spec, axis)
    }
    fn hsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>> {
        hsplit(self, spec)
    }
    fn vsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>> {
        vsplit(self, spec)
    }
    fn dsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>> {
        dsplit(self, spec)
    }
    fn unstack(&self, axis: isize) -> Result<Vec<NdArray<T>>> {
        unstack(self, axis)
    }
    fn tile(&self, reps: &[usize]) -> Result<NdArray<T>> {
        tile(self, reps)
    }
    fn repeat(&self, repeats: &[usize], axis: Option<isize>) -> Result<NdArray<T>> {
        repeat(self, repeats, axis)
    }
    fn flip_axes(&self, axis: Option<&[isize]>) -> Result<NdArray<T>> {
        flip(self, axis)
    }
    fn fliplr(&self) -> Result<NdArray<T>> {
        fliplr(self)
    }
    fn flipud(&self) -> Result<NdArray<T>> {
        flipud(self)
    }
    fn rot90(&self, k: isize, axes: (isize, isize)) -> Result<NdArray<T>> {
        rot90(self, k, axes)
    }
    fn roll(&self, shift: isize, axis: Option<isize>) -> Result<NdArray<T>> {
        roll(self, shift, axis)
    }
    fn roll_axes(&self, shifts: &[isize], axes: Option<&[isize]>) -> Result<NdArray<T>> {
        roll_axes(self, shifts, axes)
    }
    fn pad(&self, pad_width: &PadWidth, mode: &PadMode<T>) -> Result<NdArray<T>> {
        pad(self, pad_width, mode)
    }
    fn resize(&self, new_shape: &[usize]) -> Result<NdArray<T>> {
        resize(self, new_shape)
    }
    fn append(&self, values: &NdArray<T>, axis: Option<isize>) -> Result<NdArray<T>> {
        append(self, values, axis)
    }
    fn insert(&self, obj: &InsertObj<'_>, values: &NdArray<T>, axis: Option<isize>) -> Result<NdArray<T>> {
        insert(self, obj, values, axis)
    }
    fn delete(&self, obj: &DeleteObj<'_>, axis: Option<isize>) -> Result<NdArray<T>> {
        delete(self, obj, axis)
    }
    fn trim_zeros(&self, trim: TrimDirection, axis: Option<&[isize]>) -> Result<NdArray<T>> {
        trim_zeros(self, trim, axis)
    }
    fn diag(&self, k: isize) -> Result<NdArray<T>> {
        diag(self, k)
    }
    fn diagflat(&self, k: isize) -> Result<NdArray<T>> {
        diagflat(self, k)
    }
    fn tril(&self, k: isize) -> Result<NdArray<T>> {
        tril(self, k)
    }
    fn triu(&self, k: isize) -> Result<NdArray<T>> {
        triu(self, k)
    }
    fn vander(&self, n: Option<usize>, increasing: bool) -> Result<NdArray<T>> {
        vander(self, n, increasing)
    }
    fn take(&self, indices: &NdArray<i64>, axis: Option<isize>, mode: IndexMode) -> Result<NdArray<T>> {
        take(self, indices, axis, mode)
    }
    fn take_along_axis(&self, indices: &NdArray<i64>, axis: Option<isize>) -> Result<NdArray<T>> {
        take_along_axis(self, indices, axis)
    }
    fn put(&mut self, ind: &NdArray<i64>, v: &NdArray<T>, mode: IndexMode) -> Result<()> {
        put(self, ind, v, mode)
    }
    fn put_along_axis(&mut self, indices: &NdArray<i64>, values: &NdArray<T>, axis: Option<isize>) -> Result<()> {
        put_along_axis(self, indices, values, axis)
    }
    fn putmask(&mut self, mask: &NdArray<bool>, values: &NdArray<T>) -> Result<()> {
        putmask(self, mask, values)
    }
    fn compress(&self, condition: &NdArray<bool>, axis: Option<isize>) -> Result<NdArray<T>> {
        compress(condition, self, axis)
    }
    fn nonzero(&self) -> Result<Vec<NdArray<i64>>> {
        nonzero(self)
    }
    fn argwhere(&self) -> Result<NdArray<i64>> {
        argwhere(self)
    }
    fn flatnonzero(&self) -> Result<NdArray<i64>> {
        flatnonzero(self)
    }
    fn diagonal(&self, offset: isize, axis1: isize, axis2: isize) -> Result<NdArray<T>> {
        diagonal(self, offset, axis1, axis2)
    }
    fn fill_diagonal(&mut self, val: &NdArray<T>, wrap: bool) -> Result<()> {
        fill_diagonal(self, val, wrap)
    }
    fn boolean_mask(&self, mask: &NdArray<bool>) -> Result<NdArray<T>> {
        boolean_mask(self, mask)
    }
    fn boolean_mask_set(&mut self, mask: &NdArray<bool>, values: &NdArray<T>) -> Result<()> {
        boolean_mask_set(self, mask, values)
    }
    fn index_fancy(&self, indices: &[&NdArray<i64>]) -> Result<NdArray<T>> {
        index_fancy(self, indices)
    }
    fn index_fancy_set(&mut self, indices: &[&NdArray<i64>], values: &NdArray<T>) -> Result<()> {
        index_fancy_set(self, indices, values)
    }
    fn index_advanced(&self, args: &[FancyArg<'_>]) -> Result<NdArray<T>> {
        index_advanced(self, args)
    }
    fn copy_from(&mut self, src: &NdArray<T>, where_mask: Option<&NdArray<bool>>) -> Result<()> {
        copyto(self, src, where_mask)
    }
    fn shares_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool {
        shares_memory(self, other)
    }
    fn may_share_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool {
        may_share_memory(self, other)
    }
    fn array2string(&self, options: &PrintOptions) -> Result<String> {
        array2string(self, options)
    }
    fn array2string_default(&self) -> Result<String> {
        array2string_default(self)
    }
    fn array_str(&self, options: &PrintOptions) -> Result<String> {
        array_str(self, options)
    }
    fn array_repr(&self, options: &PrintOptions) -> Result<String> {
        array_repr(self, options)
    }
}

/// Extension methods provided by `numpy-core` on dynamic `DynArray`.
pub trait DynArrayExt {
    fn reshape_with_order(&self, newshape: &[isize], order: Order) -> Result<DynArray>;
    fn ravel(&self) -> Result<DynArray>;
    fn ravel_with_order(&self, order: Order) -> Result<DynArray>;
    fn flatten(&self) -> Result<DynArray>;
    fn flatten_with_order(&self, order: Order) -> Result<DynArray>;
    fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<DynArray>;
    fn moveaxis(&self, source: &[isize], destination: &[isize]) -> Result<DynArray>;
    fn rollaxis(&self, axis: isize, start: isize) -> Result<DynArray>;
    fn squeeze(&self, axis: Option<&[isize]>) -> Result<DynArray>;
    fn expand_dims(&self, axes: &[isize]) -> Result<DynArray>;
    fn atleast_1d(&self) -> Result<DynArray>;
    fn atleast_2d(&self) -> Result<DynArray>;
    fn atleast_3d(&self) -> Result<DynArray>;
    fn split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<DynArray>>;
    fn array_split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<DynArray>>;
    fn hsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<DynArray>>;
    fn vsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<DynArray>>;
    fn dsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<DynArray>>;
    fn unstack(&self, axis: isize) -> Result<Vec<DynArray>>;
    fn tile(&self, reps: &[usize]) -> Result<DynArray>;
    fn repeat(&self, repeats: &[usize], axis: Option<isize>) -> Result<DynArray>;
    fn flip(&self, axis: Option<&[isize]>) -> Result<DynArray>;
    fn fliplr(&self) -> Result<DynArray>;
    fn flipud(&self) -> Result<DynArray>;
    fn rot90(&self, k: isize, axes: (isize, isize)) -> Result<DynArray>;
    fn roll(&self, shift: isize, axis: Option<isize>) -> Result<DynArray>;
    fn roll_axes(&self, shifts: &[isize], axes: Option<&[isize]>) -> Result<DynArray>;
    fn pad(&self, pad_width: &PadWidth, mode: &DynPadMode) -> Result<DynArray>;
    fn resize(&self, new_shape: &[usize]) -> Result<DynArray>;
    fn append(&self, values: &DynArray, axis: Option<isize>) -> Result<DynArray>;
    fn insert(&self, obj: &DynInsertObj<'_>, values: &DynArray, axis: Option<isize>) -> Result<DynArray>;
    fn delete(&self, obj: &DynDeleteObj<'_>, axis: Option<isize>) -> Result<DynArray>;
    fn trim_zeros(&self, trim: TrimDirection, axis: Option<&[isize]>) -> Result<DynArray>;
    fn take(&self, indices: &DynArray, axis: Option<isize>, mode: IndexMode) -> Result<DynArray>;
    fn take_along_axis(&self, indices: &DynArray, axis: Option<isize>) -> Result<DynArray>;
    fn put(&mut self, ind: &DynArray, v: &DynArray, mode: IndexMode) -> Result<()>;
    fn put_along_axis(&mut self, indices: &DynArray, values: &DynArray, axis: Option<isize>) -> Result<()>;
    fn putmask(&mut self, mask: &DynArray, values: &DynArray) -> Result<()>;
    fn compress(&self, condition: &DynArray, axis: Option<isize>) -> Result<DynArray>;
    fn nonzero(&self) -> Result<Vec<DynArray>>;
    fn argwhere(&self) -> Result<DynArray>;
    fn flatnonzero(&self) -> Result<DynArray>;
    fn diagonal(&self, offset: isize, axis1: isize, axis2: isize) -> Result<DynArray>;
    fn fill_diagonal(&mut self, val: &DynArray, wrap: bool) -> Result<()>;
    fn boolean_mask(&self, mask: &DynArray) -> Result<DynArray>;
    fn boolean_mask_set(&mut self, mask: &DynArray, values: &DynArray) -> Result<()>;
    fn index_fancy(&self, indices: &[&DynArray]) -> Result<DynArray>;
    fn index_fancy_set(&mut self, indices: &[&DynArray], values: &DynArray) -> Result<()>;
    fn index_advanced(&self, args: &[DynFancyArg<'_>]) -> Result<DynArray>;
    fn copy_from(&mut self, src: &DynArray, where_mask: Option<&DynArray>) -> Result<()>;
    fn copy_from_casting(
        &mut self,
        src: &DynArray,
        casting: numpy_types::Casting,
        where_mask: Option<&DynArray>,
    ) -> Result<()>;
    fn shares_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool;
    fn may_share_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool;
    fn array2string(&self, options: &PrintOptions) -> Result<String>;
    fn array2string_default(&self) -> Result<String>;
    fn array_str(&self, options: &PrintOptions) -> Result<String>;
    fn array_repr(&self, options: &PrintOptions) -> Result<String>;
}

impl DynArrayExt for DynArray {
    fn reshape_with_order(&self, newshape: &[isize], order: Order) -> Result<DynArray> {
        dyn_reshape_with_order(self, newshape, order)
    }
    fn ravel(&self) -> Result<DynArray> {
        dyn_ravel(self)
    }
    fn ravel_with_order(&self, order: Order) -> Result<DynArray> {
        dyn_ravel_with_order(self, order)
    }
    fn flatten(&self) -> Result<DynArray> {
        dyn_flatten(self)
    }
    fn flatten_with_order(&self, order: Order) -> Result<DynArray> {
        dyn_flatten_with_order(self, order)
    }
    fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<DynArray> {
        dyn_swapaxes(self, axis1, axis2)
    }
    fn moveaxis(&self, source: &[isize], destination: &[isize]) -> Result<DynArray> {
        dyn_moveaxis(self, source, destination)
    }
    fn rollaxis(&self, axis: isize, start: isize) -> Result<DynArray> {
        dyn_rollaxis(self, axis, start)
    }
    fn squeeze(&self, axis: Option<&[isize]>) -> Result<DynArray> {
        dyn_squeeze(self, axis)
    }
    fn expand_dims(&self, axes: &[isize]) -> Result<DynArray> {
        dyn_expand_dims(self, axes)
    }
    fn atleast_1d(&self) -> Result<DynArray> {
        dyn_atleast_1d(self)
    }
    fn atleast_2d(&self) -> Result<DynArray> {
        dyn_atleast_2d(self)
    }
    fn atleast_3d(&self) -> Result<DynArray> {
        dyn_atleast_3d(self)
    }
    fn split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<DynArray>> {
        dyn_split(self, spec, axis)
    }
    fn array_split(&self, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<DynArray>> {
        dyn_array_split(self, spec, axis)
    }
    fn hsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<DynArray>> {
        dyn_hsplit(self, spec)
    }
    fn vsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<DynArray>> {
        dyn_vsplit(self, spec)
    }
    fn dsplit(&self, spec: SplitSpec<'_>) -> Result<Vec<DynArray>> {
        dyn_dsplit(self, spec)
    }
    fn unstack(&self, axis: isize) -> Result<Vec<DynArray>> {
        dyn_unstack(self, axis)
    }
    fn tile(&self, reps: &[usize]) -> Result<DynArray> {
        dyn_tile(self, reps)
    }
    fn repeat(&self, repeats: &[usize], axis: Option<isize>) -> Result<DynArray> {
        dyn_repeat(self, repeats, axis)
    }
    fn flip(&self, axis: Option<&[isize]>) -> Result<DynArray> {
        dyn_flip(self, axis)
    }
    fn fliplr(&self) -> Result<DynArray> {
        dyn_fliplr(self)
    }
    fn flipud(&self) -> Result<DynArray> {
        dyn_flipud(self)
    }
    fn rot90(&self, k: isize, axes: (isize, isize)) -> Result<DynArray> {
        dyn_rot90(self, k, axes)
    }
    fn roll(&self, shift: isize, axis: Option<isize>) -> Result<DynArray> {
        dyn_roll(self, shift, axis)
    }
    fn roll_axes(&self, shifts: &[isize], axes: Option<&[isize]>) -> Result<DynArray> {
        dyn_roll_axes(self, shifts, axes)
    }
    fn pad(&self, pad_width: &PadWidth, mode: &DynPadMode) -> Result<DynArray> {
        dyn_pad(self, pad_width, mode)
    }
    fn resize(&self, new_shape: &[usize]) -> Result<DynArray> {
        dyn_resize(self, new_shape)
    }
    fn append(&self, values: &DynArray, axis: Option<isize>) -> Result<DynArray> {
        dyn_append(self, values, axis)
    }
    fn insert(&self, obj: &DynInsertObj<'_>, values: &DynArray, axis: Option<isize>) -> Result<DynArray> {
        dyn_insert(self, obj, values, axis)
    }
    fn delete(&self, obj: &DynDeleteObj<'_>, axis: Option<isize>) -> Result<DynArray> {
        dyn_delete(self, obj, axis)
    }
    fn trim_zeros(&self, trim: TrimDirection, axis: Option<&[isize]>) -> Result<DynArray> {
        dyn_trim_zeros(self, trim, axis)
    }
    fn take(&self, indices: &DynArray, axis: Option<isize>, mode: IndexMode) -> Result<DynArray> {
        dyn_take(self, indices, axis, mode)
    }
    fn take_along_axis(&self, indices: &DynArray, axis: Option<isize>) -> Result<DynArray> {
        dyn_take_along_axis(self, indices, axis)
    }
    fn put(&mut self, ind: &DynArray, v: &DynArray, mode: IndexMode) -> Result<()> {
        dyn_put(self, ind, v, mode)
    }
    fn put_along_axis(&mut self, indices: &DynArray, values: &DynArray, axis: Option<isize>) -> Result<()> {
        dyn_put_along_axis(self, indices, values, axis)
    }
    fn putmask(&mut self, mask: &DynArray, values: &DynArray) -> Result<()> {
        dyn_putmask(self, mask, values)
    }
    fn compress(&self, condition: &DynArray, axis: Option<isize>) -> Result<DynArray> {
        dyn_compress(condition, self, axis)
    }
    fn nonzero(&self) -> Result<Vec<DynArray>> {
        dyn_nonzero(self)
    }
    fn argwhere(&self) -> Result<DynArray> {
        dyn_argwhere(self)
    }
    fn flatnonzero(&self) -> Result<DynArray> {
        dyn_flatnonzero(self)
    }
    fn diagonal(&self, offset: isize, axis1: isize, axis2: isize) -> Result<DynArray> {
        dyn_diagonal(self, offset, axis1, axis2)
    }
    fn fill_diagonal(&mut self, val: &DynArray, wrap: bool) -> Result<()> {
        dyn_fill_diagonal(self, val, wrap)
    }
    fn boolean_mask(&self, mask: &DynArray) -> Result<DynArray> {
        dyn_boolean_mask(self, mask)
    }
    fn boolean_mask_set(&mut self, mask: &DynArray, values: &DynArray) -> Result<()> {
        dyn_boolean_mask_set(self, mask, values)
    }
    fn index_fancy(&self, indices: &[&DynArray]) -> Result<DynArray> {
        dyn_index_fancy(self, indices)
    }
    fn index_fancy_set(&mut self, indices: &[&DynArray], values: &DynArray) -> Result<()> {
        dyn_index_fancy_set(self, indices, values)
    }
    fn index_advanced(&self, args: &[DynFancyArg<'_>]) -> Result<DynArray> {
        dyn_index_advanced(self, args)
    }
    fn copy_from(&mut self, src: &DynArray, where_mask: Option<&DynArray>) -> Result<()> {
        dyn_copyto(self, src, numpy_types::Casting::SameKind, where_mask)
    }
    fn copy_from_casting(
        &mut self,
        src: &DynArray,
        casting: numpy_types::Casting,
        where_mask: Option<&DynArray>,
    ) -> Result<()> {
        dyn_copyto(self, src, casting, where_mask)
    }
    fn shares_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool {
        shares_memory(self, other)
    }
    fn may_share_memory_with<B: MemoryInspect + ?Sized>(&self, other: &B) -> bool {
        may_share_memory(self, other)
    }
    fn array2string(&self, options: &PrintOptions) -> Result<String> {
        dyn_array2string(self, options)
    }
    fn array2string_default(&self) -> Result<String> {
        dyn_array2string_default(self)
    }
    fn array_str(&self, options: &PrintOptions) -> Result<String> {
        dyn_array_str(self, options)
    }
    fn array_repr(&self, options: &PrintOptions) -> Result<String> {
        dyn_array_repr(self, options)
    }
}
