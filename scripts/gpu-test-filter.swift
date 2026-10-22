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
if #unavailable(macOS 27) {
    fputs("MSL 4.1 acquire/release hardware tests require macOS 27.\n", stderr)
    filters.append("not test(msl41_)")
}
print(filters.isEmpty ? "all()" : filters.joined(separator: " and "))
