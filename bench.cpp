#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

static inline void black_box_u64(std::uint64_t x) {
    asm volatile("" : "+r"(x) : : "memory");
}

// ---------- ordinary branchy binary search ----------
template <class T>
static size_t lower_bound_branchy(const T* a, size_t n, T x) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ---------- branchless binary search (monobound + cmov) ----------
template <class T>
static size_t lower_bound_branchless(const T* a, size_t n, T x) {
    const T* base = a;
    size_t len = n;
    while (len > 1) {
        size_t half = len >> 1;
        base += (base[half] < x) * half;
        len -= half;
    }
    return (size_t)(base - a) + (base[0] < x);
}

// ---------- AVX2 linear brute force ----------
#if defined(__AVX2__)

static inline __m256i ge8(__m256i v, __m256i x) {
    const __m256i sign = _mm256_set1_epi8(-128);
    v = _mm256_xor_si256(v, sign);
    x = _mm256_xor_si256(x, sign);
    __m256i lt = _mm256_cmpgt_epi8(x, v);
    return _mm256_xor_si256(lt, _mm256_set1_epi8(-1));
}

static inline __m256i ge16(__m256i v, __m256i x) {
    const __m256i sign = _mm256_set1_epi16((short)0x8000);
    v = _mm256_xor_si256(v, sign);
    x = _mm256_xor_si256(x, sign);
    __m256i lt = _mm256_cmpgt_epi16(x, v);
    return _mm256_xor_si256(lt, _mm256_set1_epi16(-1));
}

static inline __m256i ge32(__m256i v, __m256i x) {
    const __m256i sign = _mm256_set1_epi32((int)0x80000000u);
    v = _mm256_xor_si256(v, sign);
    x = _mm256_xor_si256(x, sign);
    __m256i lt = _mm256_cmpgt_epi32(x, v);
    return _mm256_xor_si256(lt, _mm256_set1_epi32(-1));
}

static size_t avx2_lower_bound_u8(const u8* a, size_t n, u8 x) {
    constexpr size_t W = 32;
    __m256i xv = _mm256_set1_epi8((char)x);
    size_t i = 0;
    for (; i < n; i += W) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(a + i));
        std::uint32_t m = (std::uint32_t)_mm256_movemask_epi8(ge8(v, xv));
        if (m) {
            size_t idx = i + (size_t)__builtin_ctz(m);
            return idx < n ? idx : n;
        }
    }
    return n;
}

static size_t avx2_lower_bound_u16(const u16* a, size_t n, u16 x) {
    constexpr size_t W = 16;
    __m256i xv = _mm256_set1_epi16((short)x);
    size_t i = 0;
    for (; i < n; i += W) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(a + i));
        std::uint32_t m = (std::uint32_t)_mm256_movemask_epi8(ge16(v, xv));
        if (m) {
            size_t idx = i + (size_t)(__builtin_ctz(m) >> 1);
            return idx < n ? idx : n;
        }
    }
    return n;
}

static size_t avx2_lower_bound_u32(const u32* a, size_t n, u32 x) {
    constexpr size_t W = 8;
    __m256i xv = _mm256_set1_epi32((int)x);
    size_t i = 0;
    for (; i < n; i += W) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(a + i));
        std::uint32_t m = (std::uint32_t)_mm256_movemask_epi8(ge32(v, xv));
        if (m) {
            size_t idx = i + (size_t)(__builtin_ctz(m) >> 2);
            return idx < n ? idx : n;
        }
    }
    return n;
}

template <class T>
static size_t avx2_lower_bound(const T* a, size_t n, T x) {
    if constexpr (std::is_same_v<T, u8>) return avx2_lower_bound_u8(a, n, x);
    if constexpr (std::is_same_v<T, u16>) return avx2_lower_bound_u16(a, n, x);
    if constexpr (std::is_same_v<T, u32>) return avx2_lower_bound_u32(a, n, x);
    return 0;
}

#else
template <class T>
static size_t avx2_lower_bound(const T* a, size_t n, T x) {
    for (size_t i = 0; i < n; ++i)
        if (a[i] >= x) return i;
    return n;
}
#endif

// ---------- verification ----------
template <class T>
static bool verify_type(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr size_t W = 32 / sizeof(T);
    for (size_t n : {1u, 7u, 64u, 257u, 4096u}) {
        if constexpr (std::is_same_v<T, u8>) {
            if (n > 256) continue;
        } else if constexpr (std::is_same_v<T, u16>) {
            if (n > 65536) continue;
        }
        std::vector<T> a(n + W, (T)-1);
        for (size_t i = 0; i < n; ++i) a[i] = (T)i;
        for (size_t q = 0; q < 2000; ++q) {
            T x = (T)(rng() % (n * 2 + 1));
            size_t want = (size_t)(std::lower_bound(a.begin(), a.end(), x) - a.begin());
            size_t g1 = lower_bound_branchy(a.data(), n, x);
            size_t g2 = lower_bound_branchless(a.data(), n, x);
            size_t g3 = avx2_lower_bound(a.data(), n, x);
            if (g1 != want || g2 != want || g3 != want) {
                std::printf("VERIFY FAIL n=%zu x=%u want=%zu branchy=%zu branchless=%zu avx2=%zu\n",
                            n, (unsigned)x, want, g1, g2, g3);
                return false;
            }
        }
    }
    return true;
}

// ---------- measurement ----------
static const size_t ROUNDS = 11;

template <class T, class F>
static double bench_one(const T* a, size_t n, const std::vector<T>& xs, F f) {
    std::uint64_t acc = 0;
    acc += f(a, n, xs[0]);
    black_box_u64(acc);

    double best = 1e300;
    for (size_t r = 0; r < ROUNDS; ++r) {
        acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (T x : xs) acc += (std::uint64_t)f(a, n, x);
        auto t1 = std::chrono::steady_clock::now();
        black_box_u64(acc);
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (ns < best) best = ns;
    }
    return best / (double)xs.size();
}

template <class T>
static void run_type(const char* name) {
    const size_t max_n = std::is_same_v<T, u8> ? 256u : (std::is_same_v<T, u16> ? 65536u : 1048576u);
    constexpr size_t W = 32 / sizeof(T);
    const std::vector<size_t> all_n = {
        4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 60, 64, 96, 128, 192,
        256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 6144,
        8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072,
        196608, 262144, 393216, 524288, 786432, 1048576
    };

    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    for (size_t n : all_n) {
        if (n > max_n) continue;

        std::vector<T> a(n + W, (T)-1);
        for (size_t i = 0; i < n; ++i) a[i] = (T)i;

        size_t q_brute = std::min<size_t>(
            65536, std::max<size_t>(64, 400'000'000u / std::max<size_t>(n / 2, 1)));
        size_t q_bin = 65536;

        std::vector<T> xs_brute(q_brute), xs_bin(q_bin);
        for (size_t i = 0; i < q_brute; ++i) xs_brute[i] = a[rng() % n];
        for (size_t i = 0; i < q_bin; ++i) xs_bin[i] = a[rng() % n];

        // Present queries: expected index is the generated index itself.
        for (size_t i = 0; i < q_brute; ++i)
            if (avx2_lower_bound(a.data(), n, xs_brute[i]) != (size_t)xs_brute[i])
                std::abort();
        for (size_t i = 0; i < q_bin; ++i) {
            T x = xs_bin[i];
            if (lower_bound_branchy(a.data(), n, x) != (size_t)x ||
                lower_bound_branchless(a.data(), n, x) != (size_t)x ||
                avx2_lower_bound(a.data(), n, x) != (size_t)x)
                std::abort();
        }

        double t_brute = bench_one(a.data(), n, xs_brute,
            [](const T* p, size_t m, T x) { return avx2_lower_bound(p, m, x); });
        double t_branchless = bench_one(a.data(), n, xs_bin,
            [](const T* p, size_t m, T x) { return lower_bound_branchless(p, m, x); });
        double t_branchy = bench_one(a.data(), n, xs_bin,
            [](const T* p, size_t m, T x) { return lower_bound_branchy(p, m, x); });

        std::printf("%s,%zu,%.3f,%.3f,%.3f\n", name, n, t_brute, t_branchless, t_branchy);
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "all";
    if (which == "u8" || which == "all") {
        if (!verify_type<u8>(1)) return 1;
        run_type<u8>("u8");
    }
    if (which == "u16" || which == "all") {
        if (!verify_type<u16>(2)) return 1;
        run_type<u16>("u16");
    }
    if (which == "u32" || which == "all") {
        if (!verify_type<u32>(3)) return 1;
        run_type<u32>("u32");
    }
    return 0;
}
