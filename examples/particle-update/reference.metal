#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

struct Position { float x; float y; };
struct Motion { Position position; float velocity[2]; };
struct Particle { uchar tag; ushort flags; uint id; Motion motion; uchar color[3]; };
struct Config { float dt; Position acceleration; };

kernel void particle_reference(
    device const Particle* input [[buffer(0)]],
    device Particle* output [[buffer(1)]],
    constant Config& config [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    Particle value = input[i];
    if (value.tag != 0) {
        for (uint step = 0; step < 2; ++step) {
            value.motion.velocity[0] += config.acceleration.x * config.dt;
            value.motion.velocity[1] += config.acceleration.y * config.dt;
            value.motion.position.x += value.motion.velocity[0] * config.dt;
            value.motion.position.y += value.motion.velocity[1] * config.dt;
        }
        value.flags = ushort(value.flags + 2);
        value.color[1] = value.color[0];
    }
    output[i] = value;
}
