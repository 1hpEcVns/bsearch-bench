use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

type U8 = u8;
type U16 = u16;
type U32 = u32;

const ROUNDS: usize = 11;

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

fn bench_one<T: Copy, F: Fn(&[T], T) -> usize>(a: &[T], xs: &[T], f: F) -> f64 {
    let mut acc = 0usize;
    acc += f(a, xs[0]);
    black_box(acc);
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        acc = 0;
        let t0 = Instant::now();
        for &x in xs {
            acc += f(a, x);
        }
        let t1 = Instant::now();
        black_box(acc);
        let ns = t1.duration_since(t0).as_secs_f64() * 1e9;
        if ns < best {
            best = ns;
        }
    }
    best / xs.len() as f64
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
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed
    };

    for &n in &all_n {
        if n > max_n {
            continue;
        }
        let mut a: Vec<T> = vec![T::ZERO; n + w];
        for i in 0..n {
            a[i] = T::from_usize(i);
        }

        let q_brute = (400_000_000usize / (n / 2).max(1)).clamp(64, 65536);
        let q_bin = 65536usize;
        let mut xs_brute = Vec::with_capacity(q_brute);
        let mut xs_bin = Vec::with_capacity(q_bin);
        for _ in 0..q_brute {
            xs_brute.push(a[(rng() % n as u64) as usize]);
        }
        for _ in 0..q_bin {
            xs_bin.push(a[(rng() % n as u64) as usize]);
        }

        for &x in &xs_brute {
            assert_eq!(avx2_lower_bound(&a, n, x), x.to_usize());
        }
        for &x in &xs_bin {
            assert_eq!(lower_bound_branchy(&a[..n], x), x.to_usize());
            assert_eq!(a[..n].partition_point(|&v| v < x), x.to_usize());
            assert_eq!(avx2_lower_bound(&a, n, x), x.to_usize());
        }

        let t_brute = bench_one(&a, &xs_brute, |a, x| avx2_lower_bound(a, n, x));
        let core = &a[..n];
        let t_branchless = bench_one(core, &xs_bin, |a, x| a.partition_point(|&v| v < x));
        let t_branchy = bench_one(core, &xs_bin, lower_bound_branchy::<T>);
        println!("{name},{n},{t_brute:.3},{t_branchless:.3},{t_branchy:.3}");
    }
}

fn verify<T: BenchVal>() {
    let mut seed = 12345u64;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed
    };
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
            let x = T::from_usize((rng() % (n as u64 * 2 + 1)) as usize);
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
