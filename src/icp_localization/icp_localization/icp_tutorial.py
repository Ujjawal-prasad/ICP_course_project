import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def best_fit_transform(A, B):
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T,U.T)
    t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
    T = np.eye(m+1)
    T[:m, :m] = R
    T[:m, -1] = t.ravel()
    return T

def nearest_neighbor(src, dst):
    assert src.shape[1] == dst.shape[1]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def iterative_closest_point_visual(A,B, max_iterations=20, tolerance=0.001):
    assert A.shape[1] == B.shape[1]
    m = A.shape[1]
    
    intermediate_A = [np.copy(A)]
    intermediate_errors = []

    src_h = np.ones((m+1, A.shape[0]))
    src_h[:m, :] = np.copy(A.T)

    dst = np.copy(B)

    prev_error = float('inf')
    T_cumulative = np.identity(m+1)

    for i in range(max_iterations):
        current_src = src_h[:m,: ].T

        distances, indices = nearest_neighbor(current_src,dst)
        T_step = best_fit_transform(current_src,dst[indices, :])
        src_h = np.dot(T_step, src_h)

        intermediate_A.append(src_h[:m, :].T)

        mean_error = np.mean(distances)
        intermediate_errors.append(mean_error)

        if np.abs(prev_error) == np.abs(mean_error):
            print(f"Convergerd at iteration {i+1} with error diffrence {np.abs(prev_error - mean_error)}")

        prev_error = mean_error

        T_cumulative = np.dot(T_step, T_cumulative)

    T_final = best_fit_transform(A, src_h[:m, :].T)
    return T_final, intermediate_A, intermediate_errors, i+1

t = np.linspace(0, 2*np.pi,10)
A = np.column_stack((t, np.sin(t)))

theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array([[c,-s],[s,c]])

translation_vector = np.array([[2,0]])
np.random.seed(42) # for reproducible randomness
randomness = 0.3 * np.random.rand(10, 2)
B = np.dot(rotation_matrix, A.T).T + translation_vector + randomness
 
# --- Run ICP and get history ---
max_iter = 20
tolerance = 0.0001 # Lower tolerance for smoother convergence potentially
T_final, history_A, history_error, iters = iterative_closest_point_visual(A, B, max_iterations=max_iter, tolerance=tolerance)
 
print(f'Converged/Stopped after {iters} iterations.')
print(f'Final Mean Error: {history_error[-1]:.4f}')
print('Final Transformation:')
print(np.round(T_final, 3))

#  Create Animation ---
fig, ax = plt.subplots()
 
# Plot target points (static)
ax.scatter(B[:, 0], B[:, 1], color='blue', label='Target B', marker='x')
 
# Plot initial source points
scatter_A = ax.scatter(history_A[0][:, 0], history_A[0][:, 1], color='red', label='Source A (moving)')
title = ax.set_title(f'Iteration 0, Mean Error: N/A')
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
ax.axis('equal') # Important for visualizing rotations correctly
 
# Determine plot limits based on all points across all iterations
all_points = np.vstack([B] + history_A)
min_vals = np.min(all_points, axis=0)
max_vals = np.max(all_points, axis=0)
range_vals = max_vals - min_vals
margin = 0.1 * range_vals # Add 10% margin
ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
 
# Animation update function
def update(frame):
    # Update source points position
    scatter_A.set_offsets(history_A[frame])
    # Update title
    error_str = f"{history_error[frame-1]:.4f}" if frame > 0 else "N/A" # Error calculated *after* step
    title.set_text(f'Iteration {frame}, Mean Error: {error_str}')
    # Return the artists that were modified
    return scatter_A, title,
 
# Create the animation
# Number of frames is number of states stored (initial + iterations)
# Interval is milliseconds between frames (e.g., 500ms = 0.5s)
ani = animation.FuncAnimation(fig, update, frames=len(history_A), 
                              interval=500, blit=True, repeat=False)
 
# Display the final plot (optional, animation already shows it)
plt.figure()
plt.scatter(history_A[-1][:, 0], history_A[-1][:, 1], color='red', label='Final A')
plt.scatter(B[:, 0], B[:, 1], color='blue', label='Target B', marker='x')
plt.legend()
plt.title(f"Final Alignment after {iters} iterations")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()