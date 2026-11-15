use numpy_bench::{gen_f32, gen_f64, inspect_layout};
use numpy_types::{NdArray, Slice, SliceArg};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAlloc;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING.load(Ordering::Relaxed) {
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn measure_allocs<R>(label: &str, f: impl FnOnce() -> R) -> R {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING.store(true, Ordering::SeqCst);
    let res = f();
    TRACKING.store(false, Ordering::SeqCst);
    let ac = ALLOC_COUNT.load(Ordering::SeqCst);
    let ab = ALLOC_BYTES.load(Ordering::SeqCst);
    let dc = DEALLOC_COUNT.load(Ordering::SeqCst);
    println!(
        "{:<36} allocs={:<4} deallocs={:<4} alloc_bytes={}",
        label, ac, dc, ab
    );
    res
}

fn main() {
    println!("=== numpy-types Alignment, Strides, and Storage Sharing Inspection ===");
    for &n in &[1usize, 7, 1_000, 100_000, 10_000_000] {
        let a = NdArray::<f64>::zeros(&[n]).unwrap();
        println!("{}", inspect_layout(&format!("zeros_f64_{n}"), &a));
    }
    let f32_arr = gen_f32(&[1_000], 1);
    println!("{}", inspect_layout("gen_f32_1k", &f32_arr));

    let mat = gen_f64(&[1000, 1000], 2);
    println!("{}", inspect_layout("mat_1000x1000", &mat));

    let sliced = mat
        .slice(&[
            SliceArg::from(100isize..900isize),
            SliceArg::from(100isize..900isize),
        ])
        .unwrap();
    println!("{}", inspect_layout("slice_800x800", &sliced));
    println!(
        "  shares_storage(mat, slice_800x800) = {}",
        mat.shares_storage(&sliced)
    );

    let step2 = mat
        .slice(&[
            SliceArg::Slice(Slice::full().step_by(2)),
            SliceArg::Slice(Slice::full().step_by(2)),
        ])
        .unwrap();
    println!("{}", inspect_layout("slice_step2_500x500", &step2));

    let flipped = mat.flip(None).unwrap();
    println!("{}", inspect_layout("flip_1000x1000", &flipped));
    println!(
        "  shares_storage(mat, flip) = {}",
        mat.shares_storage(&flipped)
    );

    let transposed = mat.t();
    println!("{}", inspect_layout("transpose_1000x1000", &transposed));
    println!(
        "  shares_storage(mat, transpose) = {}",
        mat.shares_storage(&transposed)
    );

    let row = gen_f64(&[1, 1000], 3);
    let bcast = row.broadcast_to(&[1000, 1000]).unwrap();
    println!("{}", inspect_layout("broadcast_1000x1000", &bcast));
    println!(
        "  shares_storage(row, broadcast) = {}",
        row.shares_storage(&bcast)
    );

    let contig_noop = mat.to_c_contiguous();
    println!(
        "  shares_storage(mat, noop to_c_contiguous) = {}",
        mat.shares_storage(&contig_noop)
    );
    let slice_noop = mat
        .slice(&[SliceArg::from(0isize..500isize), SliceArg::from(..)])
        .unwrap();
    println!(
        "{}",
        inspect_layout("slice_first_500_rows", &slice_noop)
    );
    let slice_noop_contig = slice_noop.to_c_contiguous();
    println!(
        "  shares_storage(slice_500_rows, to_c_contiguous) = {} (slice len {} vs storage len {})",
        slice_noop.shares_storage(&slice_noop_contig),
        slice_noop.len(),
        mat.len()
    );

    let contig_t = transposed.to_c_contiguous();
    println!("{}", inspect_layout("to_c_contiguous(T)", &contig_t));

    println!("\n=== Heap Allocation Audit (per single call) ===");
    let _z = measure_allocs("zeros([100_000])", || {
        NdArray::<f64>::zeros(&[100_000]).unwrap()
    });
    let _fv = measure_allocs("from_vec(vec![0.0; 100_000])", || {
        let v = vec![0.0f64; 100_000];
        NdArray::from_vec(v)
    });
    let _sl = measure_allocs("mat.slice(100..900, 100..900)", || {
        mat.slice(&[
            SliceArg::from(100isize..900isize),
            SliceArg::from(100isize..900isize),
        ])
        .unwrap()
    });
    let _sl_step = measure_allocs("mat.slice(step -1, ..)", || {
        mat.slice(&[
            SliceArg::Slice(Slice::full().step_by(-1)),
            SliceArg::from(..),
        ])
        .unwrap()
    });
    let _tr = measure_allocs("mat.t()", || mat.t());
    let _bc = measure_allocs("row.broadcast_to([1000, 1000])", || {
        row.broadcast_to(&[1000, 1000]).unwrap()
    });
    let _rs = measure_allocs("mat.reshape([100, 10000])", || {
        mat.reshape(&[100, 10_000]).unwrap()
    });
    let _tc_noop = measure_allocs("mat.to_c_contiguous() [noop]", || {
        mat.to_c_contiguous()
    });
    let _tc_t = measure_allocs("mat.t().to_c_contiguous()", || {
        transposed.to_c_contiguous()
    });
    let _mp = measure_allocs("mat.map(|x| x * 2.0)", || mat.map(|x| x * 2.0));
}
