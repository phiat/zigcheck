const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module — the public API consumers import
    const mod = b.addModule("zcheck", .{
        .root_source_file = b.path("src/zcheck.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Tests
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run zcheck tests");
    test_step.dependOn(&run_tests.step);
}
