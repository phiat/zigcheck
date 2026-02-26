// Shrink iterators — produce progressively simpler values from a failing input.

const std = @import("std");

/// A lazy iterator that yields shrink candidates for a value of type T.
/// Each candidate is "simpler" than the original, converging toward a
/// minimal counterexample.
pub fn ShrinkIter(comptime T: type) type {
    return struct {
        const Self = @This();

        context: *anyopaque,
        nextFn: *const fn (ctx: *anyopaque) ?T,

        pub fn next(self: Self) ?T {
            return self.nextFn(self.context);
        }

        /// An empty shrinker that yields nothing.
        pub fn empty() Self {
            return .{
                .context = undefined,
                .nextFn = struct {
                    fn f(_: *anyopaque) ?T {
                        return null;
                    }
                }.f,
            };
        }
    };
}

// ── Integer shrinking ───────────────────────────────────────────────────

/// State for binary-search-toward-zero shrinking of integers.
pub fn IntShrinkState(comptime T: type) type {
    return struct {
        const Self = @This();

        target: T, // the value we're shrinking
        lo: T, // current lower bound (always 0 for unsigned)
        hi: T, // current upper bound (abs of target)
        yielded_zero: bool,
        yielded_sign_flip: bool,
        done: bool,

        pub fn init(value: T) Self {
            const info = @typeInfo(T).int;
            if (value == 0) {
                return .{
                    .target = value,
                    .lo = 0,
                    .hi = 0,
                    .yielded_zero = true,
                    .yielded_sign_flip = true,
                    .done = true,
                };
            }

            const abs_val = if (info.signedness == .signed and value < 0)
                // careful: can't negate minInt
                if (value == std.math.minInt(T)) std.math.maxInt(T) else @as(T, -value)
            else
                value;

            return .{
                .target = value,
                .lo = 0,
                .hi = abs_val,
                .yielded_zero = false,
                .yielded_sign_flip = false,
                .done = false,
            };
        }

        pub fn next(self: *Self) ?T {
            if (self.done) return null;

            // First: try zero
            if (!self.yielded_zero) {
                self.yielded_zero = true;
                return 0;
            }

            // For negative values: try the positive version
            const info = @typeInfo(T).int;
            if (info.signedness == .signed and self.target < 0 and !self.yielded_sign_flip) {
                self.yielded_sign_flip = true;
                if (self.target != std.math.minInt(T)) {
                    return -self.target;
                }
            }

            // Binary search: yield midpoint between lo and hi
            if (self.lo >= self.hi) {
                self.done = true;
                return null;
            }

            const mid = self.lo + @divFloor(self.hi - self.lo, 2);
            if (mid == self.lo) {
                self.done = true;
                return null;
            }

            // Yield mid (or -mid for negative targets)
            self.lo = mid;

            if (info.signedness == .signed and self.target < 0) {
                return -mid;
            }
            return mid;
        }

        pub fn iter(self: *Self) ShrinkIter(T) {
            return .{
                .context = @ptrCast(self),
                .nextFn = typeErasedNext,
            };
        }

        fn typeErasedNext(ctx: *anyopaque) ?T {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.next();
        }
    };
}

/// State for boolean shrinking: true shrinks to false.
pub const BoolShrinkState = struct {
    value: bool,
    done: bool,

    pub fn init(value: bool) BoolShrinkState {
        return .{
            .value = value,
            .done = value == false, // false has no shrinks
        };
    }

    pub fn next(self: *BoolShrinkState) ?bool {
        if (self.done) return null;
        self.done = true;
        return false;
    }

    pub fn iter(self: *BoolShrinkState) ShrinkIter(bool) {
        return .{
            .context = @ptrCast(self),
            .nextFn = typeErasedNext,
        };
    }

    fn typeErasedNext(ctx: *anyopaque) ?bool {
        const self: *BoolShrinkState = @ptrCast(@alignCast(ctx));
        return self.next();
    }
};

// ── Float shrinking ──────────────────────────────────────────────────

/// State for float shrinking: yield 0.0 first, then halve toward zero.
pub fn FloatShrinkState(comptime T: type) type {
    return struct {
        const Self = @This();

        target: T,
        current: T,
        yielded_zero: bool,
        done: bool,

        pub fn init(value: T) Self {
            if (value == 0.0) {
                return .{
                    .target = value,
                    .current = 0.0,
                    .yielded_zero = true,
                    .done = true,
                };
            }
            return .{
                .target = value,
                .current = value,
                .yielded_zero = false,
                .done = false,
            };
        }

        pub fn next(self: *Self) ?T {
            if (self.done) return null;

            if (!self.yielded_zero) {
                self.yielded_zero = true;
                return 0.0;
            }

            // Halve toward zero
            self.current = self.current / 2.0;

            // Stop when we're very close to zero
            if (@abs(self.current) < @as(T, 1e-10)) {
                self.done = true;
                return null;
            }

            return self.current;
        }

        pub fn iter(self: *Self) ShrinkIter(T) {
            return .{
                .context = @ptrCast(self),
                .nextFn = typeErasedNext,
            };
        }

        fn typeErasedNext(ctx: *anyopaque) ?T {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.next();
        }
    };
}

// ── Tests ───────────────────────────────────────────────────────────────

test "int shrink: 0 produces no candidates" {
    var state = IntShrinkState(i32).init(0);
    var si = state.iter();
    try std.testing.expectEqual(null, si.next());
}

test "int shrink: positive value starts with 0" {
    var state = IntShrinkState(i32).init(100);
    var si = state.iter();
    try std.testing.expectEqual(@as(i32, 0), si.next().?);
}

test "int shrink: negative value offers positive flip" {
    var state = IntShrinkState(i32).init(-42);
    var si = state.iter();
    try std.testing.expectEqual(@as(i32, 0), si.next().?); // zero first
    try std.testing.expectEqual(@as(i32, 42), si.next().?); // sign flip
}

test "int shrink: produces decreasing candidates" {
    var state = IntShrinkState(i32).init(100);
    var si = state.iter();
    _ = si.next(); // skip 0

    var prev: i32 = 0;
    while (si.next()) |val| {
        try std.testing.expect(val > prev);
        try std.testing.expect(val < 100);
        prev = val;
    }
}

test "bool shrink: true -> false" {
    var state = BoolShrinkState.init(true);
    var si = state.iter();
    try std.testing.expectEqual(false, si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "bool shrink: false -> nothing" {
    var state = BoolShrinkState.init(false);
    var si = state.iter();
    try std.testing.expectEqual(null, si.next());
}

test "float shrink: 0.0 produces no candidates" {
    var state = FloatShrinkState(f64).init(0.0);
    var si = state.iter();
    try std.testing.expectEqual(null, si.next());
}

test "float shrink: positive value starts with 0.0" {
    var state = FloatShrinkState(f64).init(1.0);
    var si = state.iter();
    try std.testing.expectEqual(@as(f64, 0.0), si.next().?);
}

test "float shrink: produces halving candidates" {
    var state = FloatShrinkState(f64).init(1.0);
    var si = state.iter();
    _ = si.next(); // skip 0.0
    try std.testing.expectEqual(@as(f64, 0.5), si.next().?);
    try std.testing.expectEqual(@as(f64, 0.25), si.next().?);
}
