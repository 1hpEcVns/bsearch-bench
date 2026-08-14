use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

type U8 = u8;
type U16 = u16;
type U32 = u32;

const ROUNDS: usize = 9;
const TARGET_NS: f64 = 3e6;
const Q_MIN: usize = 1024;
const Q_MAX: usize = 4_000_000;

fn rng_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

trait BenchVal: Copy + Ord {
    const ZERO: Self;
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;
}

impl BenchVal for u8 {
    const ZERO: Self = 0;
    fn from_usize(v: usize) -> Self { v as u8 }
    fn to_usize(self) -> usize { self as usize }
}

impl BenchVal for u16 {
    const ZERO: Self = 0;
    fn from_usize(v: usize) -> Self { v as u16 }
    fn to_usize(self) -> usize { self as usize }
}

impl BenchVal for u32 {
    const ZERO: Self = 0;
    fn from_usize(v: usize) -> Self { v as u32 }
    fn to_usize(self) -> usize { self as usize }
}

fn lower_bound_branchy<T: Ord + Copy>(a: &[T], x: T) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if a[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline(never)]
unsafe fn avx2_lower_bound_u8(a: &[u8], logical: usize, x: u8) -> usize {
    let n = logical;
    let mut i = 0usize;
    let sign = _mm256_set1_epi8(-128);
    let xv = _mm256_set1_epi8(x as i8);
    while i < n {
        let v = _mm256_loadu_si256(a.as_ptr().add(i).cast());
        let lt = _mm256_cmpgt_epi8(
            _mm256_xor_si256(xv, sign),
            _mm256_xor_si256(v, sign),
        );
        let ge = _mm256_xor_si256(lt, _mm256_set1_epi8(-1));
        let m = _mm256_movemask_epi8(ge) as u32;
        if m != 0 {
            let idx = i + m.trailing_zeros() as usize;
            return idx.min(n);
        }
        i += 32;
    }
    n
}

#[inline(never)]
unsafe fn avx2_lower_bound_u16(a: &[u16], logical: usize, x: u16) -> usize {
    let n = logical;
    let mut i = 0usize;
    let sign = _mm256_set1_epi16(i16::MIN);
    let xv = _mm256_set1_epi16(x as i16);
    while i < n {
        let v = _mm256_loadu_si256(a.as_ptr().add(i).cast());
        let lt = _mm256_cmpgt_epi16(
            _mm256_xor_si256(xv, sign),
            _mm256_xor_si256(v, sign),
        );
        let ge = _mm256_xor_si256(lt, _mm256_set1_epi16(-1));
        let m = _mm256_movemask_epi8(ge) as u32;
        if m != 0 {
            let idx = i + (m.trailing_zeros() as usize >> 1);
            return idx.min(n);
        }
        i += 16;
    }
    n
}

#[inline(never)]
unsafe fn avx2_lower_bound_u32(a: &[u32], logical: usize, x: u32) -> usize {
    let n = logical;
    let mut i = 0usize;
    let sign = _mm256_set1_epi32(i32::MIN);
    let xv = _mm256_set1_epi32(x as i32);
    while i < n {
        let v = _mm256_loadu_si256(a.as_ptr().add(i).cast());
        let lt = _mm256_cmpgt_epi32(
            _mm256_xor_si256(xv, sign),
            _mm256_xor_si256(v, sign),
        );
        let ge = _mm256_xor_si256(lt, _mm256_set1_epi32(-1));
        let m = _mm256_movemask_epi8(ge) as u32;
        if m != 0 {
            let idx = i + (m.trailing_zeros() as usize >> 2);
            return idx.min(n);
        }
        i += 8;
    }
    n
}

fn avx2_lower_bound<T: BenchVal>(a: &[T], logical: usize, x: T) -> usize {
    if std::mem::size_of::<T>() == 1 {
        unsafe { avx2_lower_bound_u8(std::slice::from_raw_parts(a.as_ptr().cast(), a.len()), logical, *(&x as *const T as *const u8)) }
    } else if std::mem::size_of::<T>() == 2 {
        unsafe { avx2_lower_bound_u16(std::slice::from_raw_parts(a.as_ptr().cast(), a.len()), logical, *(&x as *const T as *const u16)) }
    } else if std::mem::size_of::<T>() == 4 {
        unsafe { avx2_lower_bound_u32(std::slice::from_raw_parts(a.as_ptr().cast(), a.len()), logical, *(&x as *const T as *const u32)) }
    } else {
        lower_bound_branchy(a, x)
    }
}

fn make_queries<T: BenchVal>(a: &[T], n: usize, q: usize, seed: &mut u64) -> Vec<T> {
    let mut xs = Vec::with_capacity(q);
    for _ in 0..q {
        xs.push(a[((rng_next(seed) >> 32) % n as u64) as usize]);
    }
    xs
}

// METHOD 0 = AVX2 brute, 1 = branchless (partition_point), 2 = normal branchy.
#[inline(never)]
fn time_method<T: BenchVal, const METHOD: usize>(a: &[T], n: usize, xs: &[T]) -> f64 {
    let mut acc = 0usize;
    let t0 = Instant::now();
    if METHOD == 0 {
        for &x in xs {
            acc += avx2_lower_bound(a, n, x);
        }
    } else if METHOD == 1 {
        let core = &a[..n];
        for &x in xs {
            acc += core.partition_point(|&v| v < x);
        }
    } else {
        let core = &a[..n];
        for &x in xs {
            acc += lower_bound_branchy(core, x);
        }
    }
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn calibrate_q<T: BenchVal, const METHOD: usize>(
    a: &[T],
    n: usize,
    q0: usize,
    seed: &mut u64,
) -> usize {
    let xs = make_queries(a, n, q0, seed);
    let ns = time_method::<T, METHOD>(a, n, &xs);
    let per = ns / q0 as f64;
    ((TARGET_NS / per) as usize).clamp(Q_MIN, Q_MAX)
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn run_type<T: BenchVal>(name: &str) {
    let max_n = if std::mem::size_of::<T>() == 1 {
        256
    } else if std::mem::size_of::<T>() == 2 {
        65536
    } else {
        1_048_576
    };
    let w = 32 / std::mem::size_of::<T>();
    let all_n = [
        4usize, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 60, 64, 96, 128, 192,
        256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 6144,
        8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072,
        196608, 262144, 393216, 524288, 786432, 1048576,
    ];

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;

    for &n in &all_n {
        if n > max_n {
            continue;
        }
        if n % w != 0 {
            continue;
        }
        let mut a: Vec<T> = vec![T::ZERO; n + w];
        for i in 0..n {
            a[i] = T::from_usize(i);
        }

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q_brute = calibrate_q::<T, 0>(&a, n, q0_brute, &mut seed);
        let q_branchless = calibrate_q::<T, 1>(&a, n, 65536, &mut seed);
        let q_branchy = calibrate_q::<T, 2>(&a, n, 65536, &mut seed);

        let xs_brute = make_queries(&a, n, q_brute, &mut seed);
        let xs_branchless = make_queries(&a, n, q_branchless, &mut seed);
        let xs_branchy = make_queries(&a, n, q_branchy, &mut seed);

        for &x in &xs_brute {
            assert_eq!(avx2_lower_bound(&a, n, x), x.to_usize());
        }
        for &x in &xs_branchless {
            assert_eq!(lower_bound_branchy(&a[..n], x), x.to_usize());
            assert_eq!(a[..n].partition_point(|&v| v < x), x.to_usize());
            assert_eq!(avx2_lower_bound(&a, n, x), x.to_usize());
        }
        for &x in &xs_branchy {
            assert_eq!(lower_bound_branchy(&a[..n], x), x.to_usize());
            assert_eq!(a[..n].partition_point(|&v| v < x), x.to_usize());
            assert_eq!(avx2_lower_bound(&a, n, x), x.to_usize());
        }

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_branchless = Vec::with_capacity(ROUNDS);
        let mut s_branchy = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_method::<T, 0>(&a, n, &xs_brute));
            s_branchless.push(time_method::<T, 1>(&a, n, &xs_branchless));
            s_branchy.push(time_method::<T, 2>(&a, n, &xs_branchy));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_branchless = median(&mut s_branchless) / q_branchless as f64;
        let t_branchy = median(&mut s_branchy) / q_branchy as f64;
        println!("{name},{n},{t_brute:.3},{t_branchless:.3},{t_branchy:.3}");
    }
}

fn verify<T: BenchVal>() {
    let mut seed = 12345u64;
    let w = 32 / std::mem::size_of::<T>();
    for &n in &[1usize, 7, 64, 257, 4096] {
        if std::mem::size_of::<T>() == 1 && n > 256 {
            continue;
        }
        if std::mem::size_of::<T>() == 2 && n > 65536 {
            continue;
        }
        let mut a: Vec<T> = vec![T::ZERO; n + w];
        for i in 0..n {
            a[i] = T::from_usize(i);
        }
        for _ in 0..2000 {
            let x = T::from_usize(((rng_next(&mut seed) >> 32) % (n as u64 * 2 + 1)) as usize);
            let want = a[..n].partition_point(|&v| v < x);
            assert_eq!(lower_bound_branchy(&a[..n], x), want);
            assert_eq!(a[..n].partition_point(|&v| v < x), want);
            assert_eq!(avx2_lower_bound(&a, n, x), want);
        }
    }
}

fn main() {
    verify::<U8>();
    verify::<U16>();
    verify::<U32>();
    if !is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not detected; results are scalar fallback");
    }
    run_type::<U8>("u8");
    run_type::<U16>("u16");
    run_type::<U32>("u32");
}
