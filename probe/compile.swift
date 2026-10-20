import Foundation
import Metal
let path = CommandLine.arguments[1]
let device = MTLCreateSystemDefaultDevice()!
let options = MTLCompileOptions()
options.mathMode = .safe
options.mathFloatingPointFunctions = .precise
do {
    let source = try String(contentsOfFile: path, encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: options)
    let function = library.makeFunction(name: "generated")!
    let pipeline = try device.makeComputePipelineState(function: function)
    print("PASS \(path) on \(device.name), width \(pipeline.threadExecutionWidth)")
} catch {
    print("FAIL \(path) on \(device.name): \(error)")
    exit(1)
}
