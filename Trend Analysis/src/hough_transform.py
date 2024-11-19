
import numpy as np

def hough_transform_from_point(main_point, other_points, theta_resolution=1, rho_resolution=1):
    """
     Hough Transform implementation
    """
    x_main, y_main = main_point
    
    # Filter points that are to the right of the main point
    future_points = other_points[other_points[:, 0] > x_main]
    
    if len(future_points) == 0:
        return None, None, 0
        
    max_rho = int(np.hypot(future_points[:, 0].max() - x_main, 
                          np.max(np.abs(future_points[:, 1] - y_main))))
    
    rhos = np.arange(-max_rho, max_rho, rho_resolution)
    thetas = np.deg2rad(np.arange(-89, 89, theta_resolution))
    
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    distance_threshold = 5
    
    for theta_idx, theta in enumerate(thetas):
        main_rho = x_main * np.cos(theta) + y_main * np.sin(theta)
        main_rho_idx = np.argmin(np.abs(rhos - main_rho))
        
        for x, y in future_points:
            point_rho = x * np.cos(theta) + y * np.sin(theta)
            distance = abs(point_rho - main_rho)
            
            if distance < distance_threshold:
                vote_weight = 1.0 - (distance / distance_threshold)
                accumulator[main_rho_idx, theta_idx] += vote_weight
    
    max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    rho_idx, theta_idx = max_idx
    max_votes = accumulator[rho_idx, theta_idx]
    
    if max_votes > 1.5:
        return thetas[theta_idx], rhos[rho_idx], max_votes
    return None, None, 0