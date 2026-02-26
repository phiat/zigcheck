// Built-in generators for primitive types.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink = @import("shrink.zig");

/// Generator for any integer type. Produces the full range of values.
pub fn int(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("int() requires an integer type, got " ++ @typeName(T));
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
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
pub fn intRange(comptime T: type, min: T, max: T) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("intRange() requires an integer type, got " ++ @typeName(T));
    }
    _ = min;
    _ = max;
    // TODO: use intRangeAtMost with captured min/max
    // For now, delegate to full-range int generator
    return int(T);
}

/// Generator for boolean values.
pub fn boolean() Gen(bool) {
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) bool {
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
pub fn float(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .float) @compileError("float() requires a float type, got " ++ @typeName(T));
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                return rng.float(T);
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

// ── Combinators ──────────────────────────────────────────────────────

/// Always produces the same value. Shrinks to nothing.
pub fn constant(comptime T: type, comptime value: T) Gen(T) {
    return .{
        .genFn = struct {
            fn f(_: std.Random, _: std.mem.Allocator) T {
                return value;
            }
        }.f,
        .shrinkFn = struct {
            fn f(_: T, _: std.mem.Allocator) ShrinkIter(T) {
                return ShrinkIter(T).empty();
            }
        }.f,
    };
}

/// Pick uniformly from a comptime-known list of values.
/// Shrinks toward earlier elements in the list.
pub fn element(comptime T: type, comptime choices: []const T) Gen(T) {
    comptime {
        if (choices.len == 0) @compileError("element() requires at least one choice");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                const idx = rng.intRangeAtMost(usize, 0, choices.len - 1);
                return choices[idx];
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Find position of current value; shrink to earlier elements
                var current_pos: usize = choices.len;
                for (choices, 0..) |c, i| {
                    if (std.meta.eql(c, value)) {
                        current_pos = i;
                        break;
                    }
                }
                if (current_pos == 0 or current_pos == choices.len) return ShrinkIter(T).empty();

                const candidates = allocator.alloc(T, current_pos) catch return ShrinkIter(T).empty();
                for (choices[0..current_pos], 0..) |c, i| {
                    candidates[i] = c;
                }

                const State = struct {
                    items: []const T,
                    pos: usize,
                };
                const state = allocator.create(State) catch return ShrinkIter(T).empty();
                state.* = .{ .items = candidates, .pos = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            if (s.pos >= s.items.len) return null;
                            const val = s.items[s.pos];
                            s.pos += 1;
                            return val;
                        }
                    }.next,
                };
            }
        }.f,
    };
}

/// Pick from one of several generators uniformly at random.
/// Shrinks using the selected generator's shrinker, then tries earlier generators.
pub fn oneOf(comptime T: type, comptime gens: []const Gen(T)) Gen(T) {
    comptime {
        if (gens.len == 0) @compileError("oneOf() requires at least one generator");
        if (gens.len == 1) return gens[0];
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, allocator: std.mem.Allocator) T {
                const idx = rng.intRangeAtMost(usize, 0, gens.len - 1);
                inline for (gens, 0..) |g, i| {
                    if (idx == i) return g.generate(rng, allocator);
                }
                unreachable;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // We don't know which generator produced the value, so try
                // all generators' shrinkers. This over-shrinks but is correct.
                const iters = allocator.alloc(ShrinkIter(T), gens.len) catch return ShrinkIter(T).empty();
                inline for (gens, 0..) |g, i| {
                    iters[i] = g.shrink(value, allocator);
                }

                const State = struct {
                    iters_arr: []ShrinkIter(T),
                    pos: usize,
                };
                const state = allocator.create(State) catch return ShrinkIter(T).empty();
                state.* = .{ .iters_arr = iters, .pos = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            while (s.pos < s.iters_arr.len) {
                                if (s.iters_arr[s.pos].next()) |val| {
                                    return val;
                                }
                                s.pos += 1;
                            }
                            return null;
                        }
                    }.next,
                };
            }
        }.f,
    };
}

/// Transform generated values: Gen(A) -> Gen(B) via a comptime-known function.
/// Shrinks by applying the mapping to the inner generator's shrink candidates.
pub fn map(
    comptime A: type,
    comptime B: type,
    comptime inner: Gen(A),
    comptime f: *const fn (A) B,
) Gen(B) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) B {
                return f(inner.generate(rng, allocator));
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: B, allocator: std.mem.Allocator) ShrinkIter(B) {
                // We can't invert f to get the original A, so we can't use
                // inner's shrinker on value directly. Best-effort: no shrinking.
                // For proper shrink-through-map, users should use mapWithShrink
                // or the runner will still shrink at the property level.
                _ = value;
                _ = allocator;
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

/// Filter generated values. Retries up to 1000 times to find a value that
/// satisfies the predicate. If no value is found, returns the last generated value.
pub fn filter(
    comptime T: type,
    comptime inner: Gen(T),
    comptime pred: *const fn (T) bool,
) Gen(T) {
    const max_retries = 1000;
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) T {
                for (0..max_retries) |_| {
                    const val = inner.generate(rng, allocator);
                    if (pred(val)) return val;
                }
                // Last resort: return whatever we got
                return inner.generate(rng, allocator);
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Wrap inner's shrinker but skip candidates that don't pass the predicate
                const inner_iter = allocator.create(ShrinkIter(T)) catch return ShrinkIter(T).empty();
                inner_iter.* = inner.shrink(value, allocator);

                return .{
                    .context = @ptrCast(inner_iter),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const iter: *ShrinkIter(T) = @ptrCast(@alignCast(ctx));
                            while (iter.next()) |candidate| {
                                if (pred(candidate)) return candidate;
                            }
                            return null;
                        }
                    }.next,
                };
            }
        }.shrinkFn,
    };
}

/// Auto-derive a generator for any supported type via comptime reflection.
pub fn auto(comptime T: type) Gen(T) {
    return switch (@typeInfo(T)) {
        .int => int(T),
        .float => float(T),
        .bool => boolean(),
        .@"enum" => enumGen(T),
        .@"struct" => structGen(T),
        else => @compileError("zcheck.auto: unsupported type " ++ @typeName(T)),
    };
}

/// Generator that picks a random enum variant.
fn enumGen(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .@"enum") @compileError("enumGen requires an enum type");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                const fields = @typeInfo(T).@"enum".fields;
                const index = rng.intRangeAtMost(usize, 0, fields.len - 1);
                inline for (fields, 0..) |field, i| {
                    if (index == i) return @enumFromInt(field.value);
                }
                unreachable;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const fields = @typeInfo(T).@"enum".fields;
                const current_ord = @intFromEnum(value);

                // Count how many variants have a lower ordinal
                var count: usize = 0;
                inline for (fields) |field| {
                    if (field.value < current_ord) count += 1;
                }

                if (count == 0) return ShrinkIter(T).empty();

                // Allocate array of candidates with lower ordinals
                const candidates = allocator.alloc(T, count) catch return ShrinkIter(T).empty();
                var idx: usize = 0;
                inline for (fields) |field| {
                    if (field.value < current_ord) {
                        candidates[idx] = @enumFromInt(field.value);
                        idx += 1;
                    }
                }

                // Allocate iterator state
                const State = struct {
                    items: []const T,
                    pos: usize,
                };
                const state = allocator.create(State) catch return ShrinkIter(T).empty();
                state.* = .{ .items = candidates, .pos = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            if (s.pos >= s.items.len) return null;
                            const val = s.items[s.pos];
                            s.pos += 1;
                            return val;
                        }
                    }.next,
                };
            }
        }.f,
    };
}

/// Generator for structs — generates each field independently.
fn structGen(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .@"struct") @compileError("structGen requires a struct type");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, allocator: std.mem.Allocator) T {
                var result: T = undefined;
                inline for (@typeInfo(T).@"struct".fields) |field| {
                    @field(result, field.name) = auto(field.type).generate(rng, allocator);
                }
                return result;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const struct_fields = @typeInfo(T).@"struct".fields;
                if (struct_fields.len == 0) return ShrinkIter(T).empty();

                // For each field, create a ShrinkIter(T) that maps field shrinks
                // to whole-struct values. Chain them: try field 0, then field 1, etc.
                //
                // State holds: which field we're on, and for each field a type-erased
                // iterator over T. We pre-build an array of ShrinkIter(T), one per field.

                const iters = allocator.alloc(ShrinkIter(T), struct_fields.len) catch return ShrinkIter(T).empty();

                inline for (struct_fields, 0..) |field, i| {
                    // Create a mapper state that wraps field's ShrinkIter(FieldType)
                    // and produces ShrinkIter(T) by substituting the field in original.
                    const Mapper = struct {
                        original: T,
                        field_shrinker: ShrinkIter(field.type),

                        fn nextMapped(ctx: *anyopaque) ?T {
                            const self: *@This() = @ptrCast(@alignCast(ctx));
                            if (self.field_shrinker.next()) |shrunk_val| {
                                var result = self.original;
                                @field(result, field.name) = shrunk_val;
                                return result;
                            }
                            return null;
                        }
                    };
                    const mapper = allocator.create(Mapper) catch return ShrinkIter(T).empty();
                    const field_gen = auto(field.type);
                    mapper.* = .{
                        .original = value,
                        .field_shrinker = field_gen.shrink(@field(value, field.name), allocator),
                    };
                    iters[i] = .{
                        .context = @ptrCast(mapper),
                        .nextFn = Mapper.nextMapped,
                    };
                }

                // Chain state: iterate through the array of ShrinkIter(T)
                const ChainState = struct {
                    iters_arr: []ShrinkIter(T),
                    pos: usize,

                    fn nextChain(ctx: *anyopaque) ?T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        while (self.pos < self.iters_arr.len) {
                            if (self.iters_arr[self.pos].next()) |val| {
                                return val;
                            }
                            self.pos += 1;
                        }
                        return null;
                    }
                };
                const chain = allocator.create(ChainState) catch return ShrinkIter(T).empty();
                chain.* = .{ .iters_arr = iters, .pos = 0 };

                return .{
                    .context = @ptrCast(chain),
                    .nextFn = ChainState.nextChain,
                };
            }
        }.f,
    };
}

// ── Tests ───────────────────────────────────────────────────────────────

test "int generator produces values" {
    var prng = std.Random.DefaultPrng.init(12345);
    const g = int(i32);
    var seen_positive = false;
    var seen_negative = false;
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v > 0) seen_positive = true;
        if (v < 0) seen_negative = true;
    }
    try std.testing.expect(seen_positive);
    try std.testing.expect(seen_negative);
}

test "boolean generator produces both values" {
    var prng = std.Random.DefaultPrng.init(99);
    const g = boolean();
    var seen_true = false;
    var seen_false = false;
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v) seen_true = true else seen_false = true;
    }
    try std.testing.expect(seen_true);
    try std.testing.expect(seen_false);
}

test "float generator produces values in [0, 1)" {
    var prng = std.Random.DefaultPrng.init(77);
    const g = float(f64);
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v >= 0.0 and v < 1.0);
    }
}

test "enum generator produces all variants" {
    const Color = enum { red, green, blue };
    var prng = std.Random.DefaultPrng.init(42);
    const g = enumGen(Color);
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        seen[@intFromEnum(v)] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "auto: struct generation" {
    const Point = struct { x: i32, y: i32 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(Point);
    const p = g.generate(prng.random(), std.testing.allocator);
    // Just verify it produces a valid struct
    _ = p.x;
    _ = p.y;
}

test "auto: nested struct generation" {
    const Inner = struct { a: u8, b: bool };
    const Outer = struct { inner: Inner, value: i64 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(Outer);
    const v = g.generate(prng.random(), std.testing.allocator);
    _ = v.inner.a;
    _ = v.value;
}

// ── Shrink tests ─────────────────────────────────────────────────────

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

test "enum shrink: blue -> red, green" {
    const Color = enum { red, green, blue };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = enumGen(Color);
    var si = g.shrink(.blue, arena_state.allocator());
    try std.testing.expectEqual(Color.red, si.next().?);
    try std.testing.expectEqual(Color.green, si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "enum shrink: first variant has no shrinks" {
    const Color = enum { red, green, blue };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = enumGen(Color);
    var si = g.shrink(.red, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "struct shrink: first candidate shrinks first field" {
    const Point = struct { x: i32, y: i32 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = auto(Point);
    var si = g.shrink(.{ .x = 100, .y = -50 }, arena_state.allocator());
    const first = si.next().?;
    // First candidate: x shrunk to 0, y unchanged
    try std.testing.expectEqual(@as(i32, 0), first.x);
    try std.testing.expectEqual(@as(i32, -50), first.y);
}

// ── Combinator tests ─────────────────────────────────────────────────

test "constant: always produces the same value" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = constant(u32, 7);
    for (0..50) |_| {
        try std.testing.expectEqual(@as(u32, 7), g.generate(prng.random(), std.testing.allocator));
    }
}

test "element: picks from choices" {
    const choices = [_]u8{ 10, 20, 30 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = element(u8, &choices);
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v == 10) seen[0] = true;
        if (v == 20) seen[1] = true;
        if (v == 30) seen[2] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "element: shrinks toward earlier elements" {
    const choices = [_]u8{ 10, 20, 30 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = element(u8, &choices);
    var si = g.shrink(30, arena_state.allocator());
    try std.testing.expectEqual(@as(u8, 10), si.next().?);
    try std.testing.expectEqual(@as(u8, 20), si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "element: first element has no shrinks" {
    const choices = [_]u8{ 10, 20, 30 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = element(u8, &choices);
    var si = g.shrink(10, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "oneOf: picks from multiple generators" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = comptime oneOf(u32, &.{
        constant(u32, 1),
        constant(u32, 2),
        constant(u32, 3),
    });
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v == 1) seen[0] = true;
        if (v == 2) seen[1] = true;
        if (v == 3) seen[2] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "map: transforms values" {
    const g = map(u32, u64, int(u32), struct {
        fn f(x: u32) u64 {
            return @as(u64, x) * 2;
        }
    }.f);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..50) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v % 2 == 0); // always even
    }
}

test "filter: only produces values satisfying predicate" {
    const g = filter(i32, int(i32), struct {
        fn pred(n: i32) bool {
            return n > 0;
        }
    }.pred);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v > 0);
    }
}

test "filter: shrinks respect predicate" {
    const g = filter(i32, int(i32), struct {
        fn pred(n: i32) bool {
            return n >= 0;
        }
    }.pred);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var si = g.shrink(100, arena_state.allocator());
    // First candidate from int shrinker is 0, which passes pred (>= 0)
    try std.testing.expectEqual(@as(i32, 0), si.next().?);
}

test "filter: shrinks skip values failing predicate" {
    // Filter for even numbers only
    const g = filter(i32, int(i32), struct {
        fn pred(n: i32) bool {
            return @mod(n, 2) == 0;
        }
    }.pred);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var si = g.shrink(100, arena_state.allocator());
    // All shrink candidates must be even
    while (si.next()) |candidate| {
        try std.testing.expect(@mod(candidate, 2) == 0);
    }
}
