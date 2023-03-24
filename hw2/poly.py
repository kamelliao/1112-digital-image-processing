from geometrical_modification import geo_modification

# polynomial warping
def quad(x):
    return np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])

@geo_modification
def polynomial_warping(org_coord, src, dst):
    coeff = fit_polynomial(src, dst)
    quad_coord = np.apply_along_axis(quad, 0, org_coord)
    new_coord = np.dot(coeff, quad_coord).astype(int)
    return new_coord

def fit_polynomial(src, dst):
    # TODO: assert len(control points) >= degree (6)
    dst_quad = np.apply_along_axis(quad, 1, dst)  # (N, 6)
    pinv = np.linalg.pinv(dst_quad)  # (6, N)
    coeff = np.dot(pinv, src).T  # (2, 6)
    return coeff

# def control_points(img):
#     h, w = img.shape

#     # invariant
#     cx, cy = w/2-12, h/2-12
#     corners = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
#     borders = [[0, h/2], [w/2, h], [w, h/2], [w, 0]]
    
#     # to warp
#     r, R = 5, 20
#     warp_src = [[cx, cy+r], [cx-r, cy], [cx, cy-r], [cx+r, cy]]
#     warp_dst = [[cx, cy+R], [cx-R, cy], [cx, cy-R], [cx+R, cy]]
    
#     src = np.vstack([warp_src, [cx, cy], corners, borders]).astype(int)
#     dst = np.vstack([warp_dst, [cx, cy], corners, borders]).astype(int)
#     return src, dst