// Built-in generators for primitive types.
// Sub-modules handle combinators, collections, and auto-derivation.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink = @import("shrink.zig");

// -- Primitive generators -------------------------------------------------

/// Generator for any integer type. Produces the full range of values.
pub fn int(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("int() requires an integer type, got " ++ @typeName(T));
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) T {
                return rng.int(T);
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const state = allocator.create(shrink.IntShrinkState(T)) catch return ShrinkIter(T).empty();
                state.* = shrink.IntShrinkState(T).init(value);
                return state.iter();
            }
        }.f,
    };
}

/// Generator for integers in [min, max] inclusive.
pub fn intRange(comptime T: type, comptime min: T, comptime max: T) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("intRange() requires an integer type, got " ++ @typeName(T));
        if (min > max) @compileError("intRange(): min must be <= max");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) T {
                return rng.intRangeAtMost(T, min, max);
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Direct binary search in [min, value] toward min.
                // All candidates are in-range by construction — no wrapping
                // arithmetic, no silently dropped candidates.
                if (value == min) return ShrinkIter(T).empty();

                const RangeShrinkState = struct {
                    lo: T,
                    hi: T,
                    yielded_min: bool,
                    done: bool,

                    fn nextCandidate(self: *@This()) ?T {
                        if (self.done) return null;

                        if (!self.yielded_min) {
                            self.yielded_min = true;
                            return min;
                        }

                        // Binary search between lo and hi
                        if (self.lo >= self.hi) {
                            self.done = true;
                            return null;
                        }

                        // Compute midpoint safely for both signed and unsigned.
                        // For signed types, (hi - lo) can overflow the type, so
                        // we widen to a larger integer for the arithmetic.
                        const mid = blk: {
                            const bits = @typeInfo(T).int.bits;
                            const Wide = std.meta.Int(.signed, bits + 1);
                            const lo_wide: Wide = self.lo;
                            const hi_wide: Wide = self.hi;
                            const mid_wide = lo_wide + @divTrunc(hi_wide - lo_wide, 2);
                            break :blk @as(T, @intCast(mid_wide));
                        };

                        if (mid == self.lo) {
                            self.done = true;
                            return null;
                        }

                        self.lo = mid;
                        return mid;
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.nextCandidate();
                    }
                };
                const state = allocator.create(RangeShrinkState) catch return ShrinkIter(T).empty();
                state.* = .{
                    .lo = min,
                    .hi = value,
                    .yielded_min = false,
                    .done = false,
                };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = RangeShrinkState.typeErasedNext,
                };
            }
        }.f,
    };
}

/// Generator for boolean values.
pub fn boolean() Gen(bool) {
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) bool {
                return rng.boolean();
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: bool, allocator: std.mem.Allocator) ShrinkIter(bool) {
                const state = allocator.create(shrink.BoolShrinkState) catch return ShrinkIter(bool).empty();
                state.* = shrink.BoolShrinkState.init(value);
                return state.iter();
            }
        }.f,
    };
}

/// Generator for a single byte (u8).
pub fn byte() Gen(u8) {
    return int(u8);
}

/// Generator for floating point types (f16, f32, f64).
/// Produces the full range of finite float values including negatives.
/// Does not produce NaN or infinity (use floatAny for those).
pub fn float(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .float) @compileError("float() requires a float type, got " ++ @typeName(T));
    }
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) T {
                // Generate full-range finite floats via random bit patterns,
                // re-rolling NaN and infinity to keep values finite.
                while (true) {
                    const result: T = @bitCast(rng.int(Bits));
                    if (std.math.isFinite(result)) return result;
                }
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const state = allocator.create(shrink.FloatShrinkState(T)) catch return ShrinkIter(T).empty();
                state.* = shrink.FloatShrinkState(T).init(value);
                return state.iter();
            }
        }.f,
    };
}

// -- Constrained integer generators ---------------------------------------

/// Generator for strictly positive integers (> 0).
/// For unsigned types, produces [1, maxInt]. For signed types, produces [1, maxInt].
pub fn positive(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("positive() requires an integer type, got " ++ @typeName(T));
    }
    return intRange(T, 1, std.math.maxInt(T));
}

/// Generator for non-negative integers (>= 0).
/// For unsigned types this is equivalent to int(T).
/// For signed types, produces [0, maxInt].
pub fn nonNegative(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("nonNegative() requires an integer type, got " ++ @typeName(T));
    }
    return intRange(T, 0, std.math.maxInt(T));
}

/// Generator for non-zero integers (!= 0).
/// Produces any value in the full range except zero.
pub fn nonZero(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("nonZero() requires an integer type, got " ++ @typeName(T));
    }
    return combinators_mod.filter(T, int(T), struct {
        fn pred(n: T) bool {
            return n != 0;
        }
    }.pred);
}

/// Generator for strictly negative integers (< 0). Signed types only.
pub fn negative(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("negative() requires an integer type, got " ++ @typeName(T));
        if (@typeInfo(T).int.signedness != .signed) @compileError("negative() requires a signed integer type, got " ++ @typeName(T));
    }
    return intRange(T, std.math.minInt(T), -1);
}

// -- Re-exports from sub-modules ------------------------------------------

const combinators_mod = @import("combinators.zig");
const collections_mod = @import("collections.zig");
const auto_mod = @import("auto.zig");

// Combinators
pub const constant = combinators_mod.constant;
pub const element = combinators_mod.element;
pub const oneOf = combinators_mod.oneOf;
pub const map = combinators_mod.map;
pub const filter = combinators_mod.filter;
pub const FilterExhausted = combinators_mod.FilterExhausted;
pub const frequency = combinators_mod.frequency;
pub const noShrink = combinators_mod.noShrink;
pub const shrinkMap = combinators_mod.shrinkMap;
pub const flatMap = combinators_mod.flatMap;

// Collections and strings
pub const asciiChar = collections_mod.asciiChar;
pub const alphanumeric = collections_mod.alphanumeric;
pub const slice = collections_mod.slice;
pub const sliceRange = collections_mod.sliceRange;
pub const asciiString = collections_mod.asciiString;
pub const asciiStringRange = collections_mod.asciiStringRange;
pub const alphanumericString = collections_mod.alphanumericString;
pub const string = collections_mod.string;
pub const unicodeChar = collections_mod.unicodeChar;
pub const unicodeString = collections_mod.unicodeString;
pub const shuffle = collections_mod.shuffle;
pub const sublistOf = collections_mod.sublistOf;
pub const orderedList = collections_mod.orderedList;
pub const growingElements = collections_mod.growingElements;
pub const sample = collections_mod.sample;
pub const sampleWith = collections_mod.sampleWith;

// Auto-derivation
pub const auto = auto_mod.auto;

// -- Tests (pull in sub-module tests) -------------------------------------

test {
    _ = combinators_mod;
    _ = collections_mod;
    _ = auto_mod;
}

// -- Primitive tests ------------------------------------------------------

test "int generator produces values" {
    var prng = std.Random.DefaultPrng.init(12345);
    const g = int(i32);
    var seen_positive = false;
    var seen_negative = false;
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v > 0) seen_positive = true;
        if (v < 0) seen_negative = true;
    }
    try std.testing.expect(seen_positive);
    try std.testing.expect(seen_negative);
}

test "intRange: produces values in [10, 20]" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = intRange(u32, 10, 20);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v >= 10 and v <= 20);
    }
}

test "intRange: shrinker stays within [min, max]" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(u32, 10, 20);
    var si = g.shrink(15, arena_state.allocator());
    while (si.next()) |v| {
        try std.testing.expect(v >= 10);
        try std.testing.expect(v <= 20);
    }
}

test "intRange: shrinker first candidate is min" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(u32, 10, 20);
    var si = g.shrink(15, arena_state.allocator());
    const first = si.next();
    try std.testing.expect(first != null);
    try std.testing.expectEqual(@as(u32, 10), first.?);
}

test "intRange: signed range shrinker stays within bounds" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(i32, -5, 5);
    var si = g.shrink(3, arena_state.allocator());
    while (si.next()) |v| {
        try std.testing.expect(v >= -5);
        try std.testing.expect(v <= 5);
    }
}

test "intRange: signed cross-zero range produces monotonic candidates" {
    // This is the case that was problematic with the old shift-based approach:
    // intRange(i8, -100, 100) with value 50 would overflow during shift-back.
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(i8, -100, 100);
    var si = g.shrink(50, arena_state.allocator());

    // First candidate should be min (-100)
    const first = si.next().?;
    try std.testing.expectEqual(@as(i8, -100), first);

    // All subsequent candidates should be monotonically increasing toward value
    var prev: i8 = -100;
    while (si.next()) |v| {
        try std.testing.expect(v >= -100);
        try std.testing.expect(v <= 100);
        try std.testing.expect(v > prev);
        prev = v;
    }
}

test "boolean generator produces both values" {
    var prng = std.Random.DefaultPrng.init(99);
    const g = boolean();
    var seen_true = false;
    var seen_false = false;
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v) seen_true = true else seen_false = true;
    }
    try std.testing.expect(seen_true);
    try std.testing.expect(seen_false);
}

test "float generator produces finite full-range values" {
    var prng = std.Random.DefaultPrng.init(77);
    const g = float(f64);
    var seen_negative = false;
    var seen_gt_one = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(std.math.isFinite(v));
        if (v < 0.0) seen_negative = true;
        if (v > 1.0) seen_gt_one = true;
    }
    // Full-range generator should produce values outside [0,1)
    try std.testing.expect(seen_negative);
    try std.testing.expect(seen_gt_one);
}

test "int shrink: 100 first candidate is 0" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = int(i32);
    var si = g.shrink(100, arena_state.allocator());
    try std.testing.expectEqual(@as(i32, 0), si.next().?);
}

test "bool shrink: true -> false" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = boolean();
    var si = g.shrink(true, arena_state.allocator());
    try std.testing.expectEqual(false, si.next().?);
    try std.testing.expectEqual(null, si.next());
}

// -- Constrained integer generator tests ----------------------------------

test "positive: all values > 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = positive(i32);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v > 0);
    }
}

test "positive: unsigned all values > 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = positive(u32);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v > 0);
    }
}

test "nonNegative: all values >= 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = nonNegative(i32);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v >= 0);
    }
}

test "nonZero: no zeros" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = nonZero(i32);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v != 0);
    }
}

test "negative: all values < 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = negative(i32);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v < 0);
    }
}

test "positive: shrinker stays > 0" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = positive(i32);
    var si = g.shrink(50, arena_state.allocator());
    while (si.next()) |v| {
        try std.testing.expect(v > 0);
    }
}

test "negative: shrinker stays < 0" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = negative(i32);
    var si = g.shrink(-50, arena_state.allocator());
    while (si.next()) |v| {
        try std.testing.expect(v < 0);
    }
}
