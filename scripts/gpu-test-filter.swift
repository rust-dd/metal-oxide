import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("GPU tests require a Metal device.\n", stderr)
    exit(1)
}

var filters = [String]()
if !device.supportsFamily(.apple6) {
    fputs("Skipping BF16 expression graphs: \(device.name) lacks bfloat16.\n", stderr)
    filters.append("not test(generated_bfloat_expression_graphs_match_cpu)")
}
// Without these families, their tests pass by checking only that the kernels are rejected.
let rejectionOnly: [(MTLGPUFamily, String, String)] = [
    (.apple6, "Apple6", "bfloat16"),
    (.apple7, "Apple7", "float atomic, SIMD-group matrix, GEMM and shader logging"),
    (.apple8, "Apple8", "64-bit atomic min and max"),
]
for (family, name, features) in rejectionOnly where !device.supportsFamily(family) {
    fputs("\(device.name) lacks \(name): \(features) tests check only the rejection.\n", stderr)
}
if #unavailable(macOS 27) {
    fputs("MSL 4.1 acquire/release hardware tests require macOS 27.\n", stderr)
    filters.append("not test(msl41_)")
}
print(filters.isEmpty ? "all()" : filters.joined(separator: " and "))
