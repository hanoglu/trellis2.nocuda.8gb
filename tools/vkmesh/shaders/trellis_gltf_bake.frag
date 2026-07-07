#version 450

layout(location = 0) in vec3 v_position;

layout(location = 0) out vec4 out_base;
layout(location = 1) out vec4 out_mr;

layout(std430, binding = 3) readonly buffer HashEntries { int hash_entries[]; };
layout(std430, binding = 4) readonly buffer VoxelAttrs { float voxel_attrs[]; };

layout(push_constant) uniform Push {
    uint texture_size;
    uint triangle_count;
    uint voxel_count;
    uint channels;
    uint hash_size;
    uint resolution;
    uint pad0;
    uint pad1;
} pc;

uint hash_coord(ivec3 c) {
    uint h = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

int lookup_voxel(ivec3 c) {
    if (pc.hash_size == 0u) return -1;
    uint slot = hash_coord(c) & (pc.hash_size - 1u);
    for (uint probe = 0u; probe < pc.hash_size; ++probe) {
        uint off = slot * 4u;
        int value = hash_entries[off + 3u];
        if (value < 0) return -1;
        if (hash_entries[off + 0u] == c.x &&
            hash_entries[off + 1u] == c.y &&
            hash_entries[off + 2u] == c.z) {
            return value;
        }
        slot = (slot + 1u) & (pc.hash_size - 1u);
    }
    return -1;
}

vec4 fetch_base(int voxel_id) {
    vec4 pbr = vec4(0.8, 0.8, 0.8, 1.0);
    if (voxel_id < 0 || pc.channels == 0u) return pbr;
    uint base = uint(voxel_id) * pc.channels;
    if (pc.channels > 0u) pbr.x = clamp(voxel_attrs[base + 0u], 0.0, 1.0);
    if (pc.channels > 1u) pbr.y = clamp(voxel_attrs[base + 1u], 0.0, 1.0);
    if (pc.channels > 2u) pbr.z = clamp(voxel_attrs[base + 2u], 0.0, 1.0);
    if (pc.channels > 5u) pbr.w = clamp(voxel_attrs[base + 5u], 0.0, 1.0);
    return pbr;
}

float fetch_metallic(int voxel_id) {
    if (voxel_id < 0 || pc.channels <= 3u) return 0.0;
    return clamp(voxel_attrs[uint(voxel_id) * pc.channels + 3u], 0.0, 1.0);
}

float fetch_roughness(int voxel_id) {
    if (voxel_id < 0 || pc.channels <= 4u) return 0.8;
    return clamp(voxel_attrs[uint(voxel_id) * pc.channels + 4u], 0.0, 1.0);
}

int nearest_voxel(ivec3 g) {
    int best = -1;
    int res = int(pc.resolution);
    for (int radius = 0; radius <= 2 && best < 0; ++radius) {
        for (int dz = -radius; dz <= radius && best < 0; ++dz) {
            for (int dy = -radius; dy <= radius && best < 0; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (radius > 0 && abs(dx) != radius && abs(dy) != radius && abs(dz) != radius) continue;
                    ivec3 c = g + ivec3(dx, dy, dz);
                    if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= res || c.y >= res || c.z >= res) continue;
                    best = lookup_voxel(c);
                    if (best >= 0) break;
                }
            }
        }
    }
    return best;
}

void sample_pbr(vec3 p, out vec4 base, out float metallic, out float roughness) {
    int res = max(int(pc.resolution), 1);
    vec3 g = clamp((p + vec3(0.5)) * float(res), vec3(0.0), vec3(float(res - 1)));
    ivec3 g0 = ivec3(floor(g));
    ivec3 g1 = min(g0 + ivec3(1), ivec3(res - 1));
    vec3 f = g - vec3(g0);

    vec4 accum_base = vec4(0.0);
    float accum_metallic = 0.0;
    float accum_roughness = 0.0;
    float sum_w = 0.0;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                ivec3 c = ivec3(dx == 0 ? g0.x : g1.x, dy == 0 ? g0.y : g1.y, dz == 0 ? g0.z : g1.z);
                float wx = dx == 0 ? 1.0 - f.x : f.x;
                float wy = dy == 0 ? 1.0 - f.y : f.y;
                float wz = dz == 0 ? 1.0 - f.z : f.z;
                float w = wx * wy * wz;
                int id = lookup_voxel(c);
                if (id < 0) continue;
                accum_base += fetch_base(id) * w;
                accum_metallic += fetch_metallic(id) * w;
                accum_roughness += fetch_roughness(id) * w;
                sum_w += w;
            }
        }
    }
    if (sum_w > 1e-6) {
        base = accum_base / sum_w;
        metallic = accum_metallic / sum_w;
        roughness = accum_roughness / sum_w;
        return;
    }

    int id = nearest_voxel(ivec3(floor(g + vec3(0.5))));
    base = fetch_base(id);
    metallic = fetch_metallic(id);
    roughness = fetch_roughness(id);
}

void main() {
    vec4 base;
    float metallic;
    float roughness;
    sample_pbr(v_position, base, metallic, roughness);
    out_base = base;
    out_mr = vec4(0.0, roughness, metallic, 1.0);
}
