// Auto-derivation generators: auto(), enum, struct, optional, slice, union.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink = @import("shrink.zig");
const generators = @import("generators.zig");
const collections = @import("collections.zig");

/// Auto-derive a generator for any supported type via comptime reflection.
/// Supports: int, float, bool, enum, struct, optional (?T), pointer-to-slice
/// ([]const T), and tagged unions (union(enum)).
pub fn auto(comptime T: type) Gen(T) {
    return switch (@typeInfo(T)) {
        .int => generators.int(T),
        .float => generators.float(T),
        .bool => generators.boolean(),
        .@"enum" => enumGen(T),
        .@"struct" => structGen(T),
        .optional => |info| optionalGen(info.child),
        .pointer => |info| blk: {
            if (info.size == .slice and info.is_const) {
                break :blk sliceAutoGen(info.child);
            }
            @compileError("zcheck.auto: unsupported pointer type " ++ @typeName(T) ++ "; only []const T slices are supported");
        },
        .@"union" => |info| if (info.tag_type != null)
            unionGen(T)
        else
            @compileError("zcheck.auto: untagged union " ++ @typeName(T) ++ " is not supported; use union(enum) instead"),
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
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) T {
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
                const tag_val = @intFromEnum(value);

                // Find declaration index of current value
                var current_idx: usize = fields.len;
                inline for (fields, 0..) |field, i| {
                    if (field.value == tag_val) current_idx = i;
                }

                if (current_idx == 0 or current_idx == fields.len) return ShrinkIter(T).empty();

                // Allocate array of candidates with lower declaration index
                const candidates = allocator.alloc(T, current_idx) catch return ShrinkIter(T).empty();
                inline for (fields, 0..) |field, i| {
                    if (i < current_idx) {
                        candidates[i] = @enumFromInt(field.value);
                    }
                }

                // Allocate iterator state
                const State = struct {
                    items: []const T,
                    pos: usize,
                };
                const state = allocator.create(State) catch {
                    allocator.free(candidates);
                    return ShrinkIter(T).empty();
                };
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

/// Generator for structs -- generates each field independently.
fn structGen(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .@"struct") @compileError("structGen requires a struct type");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                var result: T = undefined;
                inline for (@typeInfo(T).@"struct".fields) |field| {
                    @field(result, field.name) = auto(field.type).generate(rng, allocator, size);
                }
                return result;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const struct_fields = @typeInfo(T).@"struct".fields;
                if (struct_fields.len == 0) return ShrinkIter(T).empty();

                const iters = allocator.alloc(ShrinkIter(T), struct_fields.len) catch return ShrinkIter(T).empty();

                inline for (struct_fields, 0..) |field, i| {
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

/// Generator for optional types -- 10% chance of null, 90% chance of a value.
fn optionalGen(comptime Child: type) Gen(?Child) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) ?Child {
                // ~10% null
                if (rng.intRangeAtMost(u8, 0, 9) == 0) return null;
                return auto(Child).generate(rng, allocator, size);
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: ?Child, allocator: std.mem.Allocator) ShrinkIter(?Child) {
                if (value) |v| {
                    // Try null first, then shrink the inner value
                    const State = struct {
                        yielded_null: bool,
                        inner: ShrinkIter(Child),

                        fn nextImpl(ctx: *anyopaque) ??Child {
                            const self: *@This() = @ptrCast(@alignCast(ctx));
                            if (!self.yielded_null) {
                                self.yielded_null = true;
                                return @as(?Child, null); // yield the ?Child value null
                            }
                            if (self.inner.next()) |shrunk| {
                                return @as(?Child, shrunk);
                            }
                            return @as(??Child, null); // end of iteration
                        }
                    };
                    const state = allocator.create(State) catch return ShrinkIter(?Child).empty();
                    state.* = .{
                        .yielded_null = false,
                        .inner = auto(Child).shrink(v, allocator),
                    };
                    return .{
                        .context = @ptrCast(state),
                        .nextFn = State.nextImpl,
                    };
                } else {
                    return ShrinkIter(?Child).empty();
                }
            }
        }.f,
    };
}

/// Generator for []const T slices via auto -- delegates to slice(T, auto(T), 20).
fn sliceAutoGen(comptime Child: type) Gen([]const Child) {
    return collections.slice(Child, auto(Child), 20);
}

/// Generator for tagged unions -- picks a random variant, generates its payload.
fn unionGen(comptime T: type) Gen(T) {
    const info = @typeInfo(T).@"union";
    const fields = info.fields;
    comptime {
        if (fields.len == 0) @compileError("unionGen: union has no fields");
    }
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                const idx = rng.intRangeAtMost(usize, 0, fields.len - 1);
                inline for (fields, 0..) |field, i| {
                    if (idx == i) {
                        if (field.type == void) {
                            return @unionInit(T, field.name, {});
                        } else {
                            return @unionInit(T, field.name, auto(field.type).generate(rng, allocator, size));
                        }
                    }
                }
                unreachable;
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Shrink by trying earlier variants (void payload), then shrinking the payload.
                const tag = @intFromEnum(std.meta.activeTag(value));

                // Count: earlier void variants + payload shrinks
                var count: usize = 0;
                inline for (fields, 0..) |field, i| {
                    if (@as(usize, @intCast(@intFromEnum(@field(info.tag_type.?, field.name)))) < tag) {
                        if (field.type == void) count += 1;
                    }
                    _ = i;
                }

                // For simplicity, just try earlier void variants
                if (count == 0) {
                    // Try shrinking the payload of the current variant
                    return shrinkUnionPayload(T, value, allocator);
                }

                const candidates = allocator.alloc(T, count) catch return ShrinkIter(T).empty();
                var pos: usize = 0;
                inline for (fields) |field| {
                    if (@as(usize, @intCast(@intFromEnum(@field(info.tag_type.?, field.name)))) < tag) {
                        if (field.type == void) {
                            candidates[pos] = @unionInit(T, field.name, {});
                            pos += 1;
                        }
                    }
                }

                const State = struct {
                    items: []const T,
                    idx: usize,
                    payload_shrinker: ShrinkIter(T),
                    payload_done: bool,

                    fn nextVariant(ctx: *anyopaque) ?T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        if (self.idx < self.items.len) {
                            const val = self.items[self.idx];
                            self.idx += 1;
                            return val;
                        }
                        if (!self.payload_done) {
                            if (self.payload_shrinker.next()) |val| return val;
                            self.payload_done = true;
                        }
                        return null;
                    }
                };
                const state = allocator.create(State) catch return ShrinkIter(T).empty();
                state.* = .{
                    .items = candidates,
                    .idx = 0,
                    .payload_shrinker = shrinkUnionPayload(T, value, allocator),
                    .payload_done = false,
                };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.nextVariant,
                };
            }

            fn shrinkUnionPayload(comptime U: type, value: U, allocator: std.mem.Allocator) ShrinkIter(U) {
                const u_info = @typeInfo(U).@"union";
                const u_fields = u_info.fields;
                inline for (u_fields) |field| {
                    if (std.meta.activeTag(value) == @field(u_info.tag_type.?, field.name)) {
                        if (field.type == void) return ShrinkIter(U).empty();
                        const payload = @field(value, field.name);
                        const inner_iter = allocator.create(ShrinkIter(field.type)) catch return ShrinkIter(U).empty();
                        inner_iter.* = auto(field.type).shrink(payload, allocator);

                        const Mapper = struct {
                            inner: *ShrinkIter(field.type),

                            fn nextMapped(ctx: *anyopaque) ?U {
                                const self: *@This() = @ptrCast(@alignCast(ctx));
                                if (self.inner.next()) |shrunk_payload| {
                                    return @unionInit(U, field.name, shrunk_payload);
                                }
                                return null;
                            }
                        };
                        const mapper = allocator.create(Mapper) catch return ShrinkIter(U).empty();
                        mapper.* = .{ .inner = inner_iter };
                        return .{
                            .context = @ptrCast(mapper),
                            .nextFn = Mapper.nextMapped,
                        };
                    }
                }
                return ShrinkIter(U).empty();
            }
        }.f,
    };
}

// -- Tests ----------------------------------------------------------------

test "enum generator produces all variants" {
    const Color = enum { red, green, blue };
    var prng = std.Random.DefaultPrng.init(42);
    const g = enumGen(Color);
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        seen[@intFromEnum(v)] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "auto: struct generation" {
    const Point = struct { x: i32, y: i32 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(Point);
    const p = g.generate(prng.random(), std.testing.allocator, 100);
    // Just verify it produces a valid struct
    _ = p.x;
    _ = p.y;
}

test "auto: nested struct generation" {
    const Inner = struct { a: u8, b: bool };
    const Outer = struct { inner: Inner, value: i64 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(Outer);
    const v = g.generate(prng.random(), std.testing.allocator, 100);
    _ = v.inner.a;
    _ = v.value;
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

test "enum shrink: sparse enum uses declaration order" {
    const Sparse = enum(u32) { a = 0, b = 100, c = 1 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = enumGen(Sparse);
    // .c has tag value 1 but declaration index 2 -- should shrink to .a and .b
    var si = g.shrink(.c, arena_state.allocator());
    try std.testing.expectEqual(Sparse.a, si.next().?);
    try std.testing.expectEqual(Sparse.b, si.next().?);
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

test "auto: optional generates both null and non-null" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(?i32);
    var seen_null = false;
    var seen_value = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v) |_| {
            seen_value = true;
        } else {
            seen_null = true;
        }
    }
    try std.testing.expect(seen_null);
    try std.testing.expect(seen_value);
}

test "auto: optional shrinks to null first" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = auto(?i32);
    var si = g.shrink(@as(?i32, 42), arena_state.allocator());
    const first = si.next().?;
    try std.testing.expectEqual(@as(?i32, null), first);
}

test "auto: slice []const i32 generates values" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto([]const i32);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(v.len <= 20);
    }
}

test "auto: tagged union generates all variants" {
    const MyUnion = union(enum) {
        empty: void,
        value: i32,
        flag: bool,
    };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(MyUnion);
    var seen_empty = false;
    var seen_value = false;
    var seen_flag = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        switch (v) {
            .empty => seen_empty = true,
            .value => seen_value = true,
            .flag => seen_flag = true,
        }
    }
    try std.testing.expect(seen_empty);
    try std.testing.expect(seen_value);
    try std.testing.expect(seen_flag);
}

test "auto: tagged union shrinker produces candidates" {
    const MyUnion = union(enum) {
        empty: void,
        value: i32,
    };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = auto(MyUnion);
    // Shrink .value(100) -- should try .empty first (earlier variant), then shrink the i32
    var si = g.shrink(MyUnion{ .value = 100 }, arena_state.allocator());
    const first = si.next().?;
    // First candidate should be the earlier void variant
    switch (first) {
        .empty => {},
        .value => return error.TestUnexpectedResult,
    }
}

// -- Shrink no-loop guarantee tests (QuickCheck parity) -------------------

fn assertNoShrinkLoop(comptime T: type, gen: Gen(T), value: T, allocator: std.mem.Allocator) !void {
    var current = value;
    for (0..50) |_| {
        var si = gen.shrink(current, allocator);
        var found_smaller = false;
        while (si.next()) |candidate| {
            if (std.meta.eql(candidate, value)) {
                return error.ShrinkLoopDetected;
            }
            if (!found_smaller) {
                current = candidate;
                found_smaller = true;
            }
        }
        if (!found_smaller) break;
    }
}

test "shrink no-loop: u32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.int(u32);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        try assertNoShrinkLoop(u32, g, v, arena.allocator());
    }
}

test "shrink no-loop: i32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.int(i32);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        try assertNoShrinkLoop(i32, g, v, arena.allocator());
    }
}

test "shrink no-loop: bool" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.boolean();
    try assertNoShrinkLoop(bool, g, true, arena.allocator());
    try assertNoShrinkLoop(bool, g, false, arena.allocator());
}

test "shrink no-loop: f64" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.float(f64);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        try assertNoShrinkLoop(f64, g, v, arena.allocator());
    }
}

test "shrink no-loop: enum" {
    const Color = enum { red, green, blue };
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = auto(Color);
    try assertNoShrinkLoop(Color, g, .blue, arena.allocator());
    try assertNoShrinkLoop(Color, g, .green, arena.allocator());
    try assertNoShrinkLoop(Color, g, .red, arena.allocator());
}

test "shrink no-loop: struct" {
    const Point = struct { x: i32, y: i32 };
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = auto(Point);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..10) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        try assertNoShrinkLoop(Point, g, v, arena.allocator());
    }
}

test "shrink no-loop: optional" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = auto(?i32);
    try assertNoShrinkLoop(?i32, g, @as(?i32, 42), arena.allocator());
    try assertNoShrinkLoop(?i32, g, @as(?i32, null), arena.allocator());
}

test "shrink no-loop: intRange" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.intRange(u32, 10, 20);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        try assertNoShrinkLoop(u32, g, v, arena.allocator());
    }
}

// -- Float shrink candidate existence tests (QuickCheck parity) -----------

test "float shrink: non-zero finite f64 always has candidates" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.float(f64);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        if (v == 0.0) continue;
        var si = g.shrink(v, arena.allocator());
        // Must have at least one shrink candidate (0.0)
        const first = si.next();
        try std.testing.expect(first != null);
    }
}

test "float shrink: non-zero finite f32 always has candidates" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = generators.float(f32);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena.allocator(), 100);
        if (v == 0.0) continue;
        var si = g.shrink(v, arena.allocator());
        const first = si.next();
        try std.testing.expect(first != null);
    }
}

// -- Shrink boundary precision tests (QuickCheck parity) ------------------

test "shrink boundary: signed n < -5 shrinks to -6" {
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, i32, generators.int(i32), struct {
        fn prop(n: i32) !void {
            if (n < -5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(@as(i32, -6), f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: intRange(10,20) value shrinks to boundary" {
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, u32, generators.intRange(u32, 10, 20), struct {
        fn prop(n: u32) !void {
            if (n > 15) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(@as(u32, 16), f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: bool property shrinks true to exact boundary" {
    const runner = @import("runner.zig");
    // Property: fail on true. Should shrink to true (it's already minimal).
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, bool, generators.boolean(), struct {
        fn prop(b: bool) !void {
            if (b) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(true, f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: enum shrinks to first failing variant" {
    const Color = enum { red, green, blue };
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, Color, auto(Color), struct {
        fn prop(c: Color) !void {
            // Fails for green and blue -- should shrink to green (second variant)
            if (c != .red) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(Color.green, f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: struct shrinks fields independently" {
    const Point = struct { x: u32, y: u32 };
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, Point, auto(Point), struct {
        fn prop(p: Point) !void {
            if (p.x >= 5 and p.y >= 5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 5), f.shrunk.x);
            try std.testing.expectEqual(@as(u32, 5), f.shrunk.y);
        },
        else => return error.TestUnexpectedResult,
    }
}
