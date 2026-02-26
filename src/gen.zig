// Gen(T) -- the core generator type.
//
// A Gen(T) knows how to produce random values of type T and how to shrink
// a failing value toward a minimal counterexample.

const std = @import("std");
const ShrinkIter = @import("shrink.zig").ShrinkIter;

/// A generator for values of type T.
///
/// Generators are composable: use `map`, `filter`, and other combinators
/// to build complex generators from simple ones.
///
/// The `size` parameter controls the "magnitude" of generated values.
/// The runner threads size linearly from 0 to 100 across test cases,
/// so early tests use small values and later tests use large ones.
/// Generators that don't use size simply ignore it.
pub fn Gen(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Generate a random value. May allocate (slices, strings).
        /// `size` grows from 0 to 100 across a test run — use it to
        /// control the magnitude of generated values.
        genFn: *const fn (rng: std.Random, allocator: std.mem.Allocator, size: usize) T,

        /// Produce shrink candidates for a value. The returned iterator
        /// yields progressively simpler values. The allocator is used to
        /// heap-allocate mutable shrink state.
        shrinkFn: *const fn (value: T, allocator: std.mem.Allocator) ShrinkIter(T),

        /// Generate a random value with the given size parameter.
        pub fn generate(self: Self, rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
            return self.genFn(rng, allocator, size);
        }

        /// Get an iterator of shrink candidates for the given value.
        pub fn shrink(self: Self, value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
            return self.shrinkFn(value, allocator);
        }
    };
}

// -- Tests ----------------------------------------------------------------

test "Gen: basic construction and generation" {
    const g = Gen(u32){
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) u32 {
                return rng.int(u32);
            }
        }.f,
        .shrinkFn = struct {
            fn f(_: u32, _: std.mem.Allocator) ShrinkIter(u32) {
                return ShrinkIter(u32).empty();
            }
        }.f,
    };

    var prng = std.Random.DefaultPrng.init(42);
    const val = g.generate(prng.random(), std.testing.allocator, 100);
    // Just verify it doesn't crash and produces a value
    _ = val;
}
