// Gen(T) — the core generator type.
//
// A Gen(T) knows how to produce random values of type T and how to shrink
// a failing value toward a minimal counterexample.

const std = @import("std");
const ShrinkIter = @import("shrink.zig").ShrinkIter;

/// A generator for values of type T.
///
/// Generators are composable: use `map`, `filter`, and other combinators
/// to build complex generators from simple ones.
pub fn Gen(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Generate a random value. May allocate (slices, strings).
        /// Caller owns any allocated memory via the provided allocator.
        genFn: *const fn (rng: std.Random, allocator: std.mem.Allocator) T,

        /// Produce shrink candidates for a value. The returned iterator
        /// yields progressively simpler values. The allocator is used to
        /// heap-allocate mutable shrink state.
        shrinkFn: *const fn (value: T, allocator: std.mem.Allocator) ShrinkIter(T),

        /// Generate a random value.
        pub fn generate(self: Self, rng: std.Random, allocator: std.mem.Allocator) T {
            return self.genFn(rng, allocator);
        }

        /// Get an iterator of shrink candidates for the given value.
        pub fn shrink(self: Self, value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
            return self.shrinkFn(value, allocator);
        }
    };
}

// ── Tests ───────────────────────────────────────────────────────────────

test "Gen: basic construction and generation" {
    const g = Gen(u32){
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) u32 {
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
    const val = g.generate(prng.random(), std.testing.allocator);
    // Just verify it doesn't crash and produces a value
    _ = val;
}
