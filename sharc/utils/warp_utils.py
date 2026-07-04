import warp as wp
import numpy as np

# --- THE HELPER CLASS ---
class WarpMeshSampler:
    def __init__(self, vertices, faces):
        self.device = 'cuda'
        print("Uploading mesh to GPU and building BVH (Once)...")
        self.wp_points = wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.wp_indices = wp.array(faces.astype(np.int32).flatten(), dtype=int, device=self.device)
        self.mesh = wp.Mesh(points=self.wp_points, indices=self.wp_indices, velocities=None)
        print("BVH Ready.")

    def query_inside(self, points_np):
        """Parity Check: Returns mask of points inside the mesh."""
        n = len(points_np)
        if n == 0: return np.array([], dtype=bool)
        
        points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        mask_wp = wp.zeros(n, dtype=int, device=self.device)
        
        wp.launch(
            kernel=robust_parity_check_kernel,
            dim=n,
            inputs=[self.mesh.id, points_wp, mask_wp, 1e-4],
            device=self.device
        )
        return mask_wp.numpy().astype(bool)

    def filter_too_close(self, points_np, min_dist):
        """Distance Check: Returns mask of points that are FAR ENOUGH from surface."""
        n = len(points_np)
        if n == 0: return np.array([], dtype=bool)
        if min_dist <= 0: return np.ones(n, dtype=bool)

        points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        mask_wp = wp.zeros(n, dtype=int, device=self.device)

        # Launch the distance kernel
        wp.launch(
            kernel=distance_filter_kernel,
            dim=n,
            inputs=[self.mesh.id, points_wp, mask_wp, min_dist],
            device=self.device
        )
        
        return mask_wp.numpy().astype(bool)

def launch_ray_cast(wp_mesh,
                    origin,
                    directions_wp,
                    distances_wp,
                    num_samples,
                    max_distance=2.0,
                    device='cuda'):

    # The origin is a single point shared by all rays; pass it as a kernel
    # scalar (wp.vec3) instead of tiling it into an (N, 3) array and uploading
    # it every call. This removes a per-reference-point CPU tile + host->device
    # copy from the hot fitting loop.
    origin_v = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))

    # Launch the optimized kernel
    wp.launch(
        kernel=optimized_ray_cast_kernel,
        dim=num_samples,
        inputs=[wp_mesh.id, origin_v, directions_wp, distances_wp, max_distance],
        device=device
    )
    wp.synchronize()
    distances = distances_wp.numpy()
    
    # Filter valid hits
    mask = (distances > 0) & (distances < max_distance) & np.isfinite(distances)
    final_distances = np.zeros_like(distances)
    final_distances[mask] = distances[mask]
    
    return final_distances

@wp.kernel
def visibility_matrix_kernel(
    mesh_id: wp.uint64,
    cands: wp.array(dtype=wp.vec3),
    surf: wp.array(dtype=wp.vec3),
    n_surf: int,
    thr: float,
    eps: float,
    out: wp.array(dtype=wp.int8),
):
    # One thread per (candidate, surface-sample) pair.
    ci, si = wp.tid()
    o = cands[ci]
    s = surf[si]
    v = s - o
    td = wp.length(v)

    covered = wp.int8(0)
    # Proximity gate first (cheap) so out-of-range pairs skip the ray query.
    if td <= thr and td > 0.0:
        d = v / td
        # Cast toward the surface point, stopping just short of it. If no
        # triangle is hit before then, the candidate has clear line-of-sight.
        q = wp.mesh_query_ray(mesh_id, o, d, td - eps)
        if not q.result:
            covered = wp.int8(1)
    out[ci * n_surf + si] = covered


@wp.kernel
def nearest_point_dist_kernel(
    cands: wp.array(dtype=wp.vec3),
    surf: wp.array(dtype=wp.vec3),
    n_surf: int,
    out: wp.array(dtype=float),
):
    # One thread per candidate: brute-force distance to its nearest surface
    # sample. Exactly equivalent to cdist(cands, surf).min(axis=1) but runs on
    # the GPU with no (n_cand x n_surf) matrix materialized.
    ci = wp.tid()
    o = cands[ci]
    best = float(1.0e30)
    for si in range(n_surf):
        dv = surf[si] - o
        d2 = wp.dot(dv, dv)
        if d2 < best:
            best = d2
    out[ci] = wp.sqrt(best)


def nearest_surface_distance_gpu(candidates, sampled_points, device='cuda'):
    """distance from each candidate to its nearest surface sample, on GPU.

    Equivalent to cdist(candidates, sampled_points).min(axis=1) but avoids
    materializing the full (n_cand, n_surf) distance matrix on CPU.
    """
    cands = np.ascontiguousarray(candidates, dtype=np.float32)
    surf = np.ascontiguousarray(sampled_points, dtype=np.float32)
    n, m = len(cands), len(surf)
    cw = wp.array(cands, dtype=wp.vec3, device=device)
    sw = wp.array(surf, dtype=wp.vec3, device=device)
    out = wp.zeros(n, dtype=float, device=device)
    wp.launch(kernel=nearest_point_dist_kernel, dim=n,
              inputs=[cw, sw, m, out], device=device)
    wp.synchronize()
    return out.numpy()


def compute_visibility_matrix_gpu(vertices, faces, sampled_points,
                                  init_skeleton_points, proximity_threshold,
                                  epsilon=1e-5, device='cuda'):
    """GPU visibility matrix via Warp BVH ray-casting.

    Returns an (m, n) boolean matrix D (surface x candidates) matching the
    semantics of the Open3D reference implementation: D[j, i] is True iff
    candidate i has line-of-sight to surface sample j AND is within
    proximity_threshold. ~200x faster than the per-candidate Open3D loop.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32).flatten()
    cands = np.ascontiguousarray(init_skeleton_points, dtype=np.float32)
    surf = np.ascontiguousarray(sampled_points, dtype=np.float32)
    n, m = len(cands), len(surf)

    mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device=device),
                   indices=wp.array(tris, dtype=int, device=device),
                   velocities=None)
    cw = wp.array(cands, dtype=wp.vec3, device=device)
    sw = wp.array(surf, dtype=wp.vec3, device=device)
    out = wp.zeros(n * m, dtype=wp.int8, device=device)

    wp.launch(
        kernel=visibility_matrix_kernel,
        dim=(n, m),
        inputs=[mesh.id, cw, sw, m, float(proximity_threshold), float(epsilon), out],
        device=device,
    )
    wp.synchronize()
    # Kernel writes row-major (candidate, surface); reshape (n, m) then
    # transpose to the (m, n) convention the caller expects.
    return out.numpy().reshape(n, m).T.astype(bool)


@wp.kernel
def distance_filter_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    valid_mask: wp.array(dtype=int),
    min_dist: float
):
    i = wp.tid()
    p = points[i]
    
    # "Is there any triangle within min_dist of this point?"
    # If yes (result=True), the point is too close -> Invalid (0)
    # If no (result=False), the point is safe -> Valid (1)
    
    query = wp.mesh_query_point(mesh_id, p, min_dist)
    
    if query.result:
        valid_mask[i] = 0 # Too close
    else:
        valid_mask[i] = 1 # Far enough

@wp.kernel
def robust_parity_check_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    inside_mask: wp.array(dtype=int),
    epsilon: float
):
    i = wp.tid()
    p = points[i]
    
    votes = int(0)
    
    # 3 Orthogonal Rays
    dirs = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    for d in range(3):
        direction = dirs[d]
        t_current = float(0.0)
        hit_count = int(0)
        
        # Max 100 bounces to prevent infinite loops
        for safe in range(100): 
            query = wp.mesh_query_ray(mesh_id, p + direction * t_current, direction, 1.0e6)
            
            if not query.result:
                break
            
            hit_count += 1
            t_current += query.t + epsilon
        
        if hit_count % 2 == 1:
            votes += 1

    if votes >= 2:
        inside_mask[i] = 1
    else:
        inside_mask[i] = 0
        
@wp.kernel
def optimized_ray_cast_kernel(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_directions: wp.array(dtype=wp.vec3),
    hit_distances: wp.array(dtype=float),
    max_dist: float
):
    i = wp.tid()

    # wp.mesh_query_ray returns true if hit, plus t, u, v, sign, normal, face_index
    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_directions[i], max_dist)

    if query.result:
        hit_distances[i] = query.t
    else:
        hit_distances[i] = -1.0