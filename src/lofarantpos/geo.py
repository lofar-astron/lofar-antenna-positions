"""Functions for geographic transformations commonly used for LOFAR"""

from numpy import sqrt, sin, cos, arctan2, array, cross, dot, float64, vstack, transpose, shape, concatenate, zeros_like, newaxis, stack, moveaxis, set_printoptions
from numpy.linalg.linalg import norm


def normalized_earth_radius(latitude_rad):
    """Compute the normalized radius of the WGS84 ellipsoid at a given latitude"""
    wgs84_f = 1. / 298.257223563
    return 1.0 / sqrt(cos(latitude_rad) ** 2 + ((1.0 - wgs84_f) ** 2) * (sin(latitude_rad) ** 2))


def geographic_from_xyz(xyz_m):
    """Compute longitude, latitude and height (from the WGS84 ellipsoid) of a given point or list
       of points

    Args:
        xyz_m (Union[array, list]): xyz-coordinates (in m) of the given point.

    Returns:
        Dict[Union[array, float]: Dictionary with 'lon_rad', 'lat_rad', 'height_m',
                                     values are float for a single input, arrays for multiple inputs

    Examples:
        >>> from pprint import pprint
        >>> xyz_m = [3836811, 430299, 5059823]
        >>> pprint(geographic_from_xyz(xyz_m))
        {'height_m': -0.28265954554080963,
        ...'lat_rad': 0.9222359279580563,
        ...'lon_rad': 0.11168348969295486}

        >>> xyz2_m = array([3828615, 438754, 5065265])
        >>> pprint(geographic_from_xyz([xyz_m, xyz2_m]))
        {'height_m': array([-0.28265955, -0.74483879]),
        ...'lat_rad': array([0.92223593, 0.92365033]),
        ...'lon_rad': array([0.11168349, 0.11410087])}

        >>> geographic_from_xyz([[xyz_m, xyz2_m]])['lon_rad'].shape
        (1, 2)
    """
    lon_rad, lat_rad, height_m = moveaxis(geographic_array_from_xyz(xyz_m), -1, 0)
    return {'lon_rad': lon_rad, 'lat_rad': lat_rad, 'height_m': height_m}


def geographic_array_from_xyz(xyz_m):
    """
    xyz_m is a (N,3) array.
    Compute lon, lat, and height
    Output an (N, 3) array with latitude (rad), longitude (rad) and height (m)

    Examples:
        >>> xyz_m = [3836811, 430299, 5059823]
        >>> geographic_array_from_xyz([xyz_m, xyz_m])
        array([[ 0.11168349,  0.92223593, -0.28265955],
               [ 0.11168349,  0.92223593, -0.28265955]])

        >>> geographic_array_from_xyz([[xyz_m, xyz_m]]).shape
        (1, 2, 3)
    """
    wgs84_a = 6378137.0
    wgs84_f = 1./298.257223563
    wgs84_e2 = wgs84_f*(2.0 - wgs84_f)
    
    x_m, y_m, z_m = moveaxis(array(xyz_m), -1, 0)
    lon_rad = arctan2(y_m, x_m)
    r_m = sqrt(x_m**2 + y_m**2)
    # Iterate to latitude solution
    phi_previous = 1e4
    phi = arctan2(z_m, r_m)
    while (abs(phi -phi_previous) > 1.6e-12).any():
        phi_previous = phi
        phi = arctan2(z_m + wgs84_e2*wgs84_a*normalized_earth_radius(phi)*sin(phi),
                      r_m)
    lat_rad = phi
    height_m = r_m*cos(lat_rad) + z_m*sin(lat_rad) - wgs84_a*sqrt(1.0 - wgs84_e2*sin(lat_rad)**2)
    return stack([lon_rad, lat_rad, height_m], -1)


def localnorth_to_etrs(centerxyz_m):
    """
    Compute a matrix that transforms from a local coordinate system tangent
    to the WGS84 ellipsoid to ETRS89 ECEF XYZ coordinates.

    Args:
        centerxyz_m (array): xyz-coordinates of the center of the local coordinate system

    Returns:
        array: 3x3 rotation matrix

    Example:
        >>> set_printoptions(suppress=True)
        >>> station1_etrs = [3801633.868, -529022.268, 5076996.892]
        >>> station2_etrs = [3826577.462,  461022.624, 5064892.526]
        >>> localnorth_to_etrs(array(station1_etrs))
        array([[ 0.13782846, -0.79200355,  0.59475516],
               [ 0.99045611,  0.11021248, -0.08276408],
               [ 0.        ,  0.60048613,  0.79963517]])
        >>> localnorth_to_etrs([[station1_etrs, station2_etrs]]).shape
        (1, 2, 3, 3)
    """
    center_lonlat = geographic_from_xyz(centerxyz_m)
    ellipsoid_normal = normal_vector_ellipsoid(center_lonlat['lon_rad'], center_lonlat['lat_rad'])
    return projection_matrix(centerxyz_m, ellipsoid_normal)


def xyz_from_geographic(lon_rad, lat_rad, height_m):
    """Compute cartesian xyz coordinates from a longitude, latitude and height (from
    the WGS84 ellipsoid)

    Args:
        lon_rad (Union[float, array]): longitude in radians
        lat_rad (Union[float, array]): latitude in radians
        height_m (Union[float, array]): height in meters

    Returns:
        array: xyz coordinates in meters

    Examples:
        >>> xyz_from_geographic(-0.1382, 0.9266, 99.115)
        array([3802111.62491437, -528822.82583168, 5076662.15079859])
        >>> coords = array([[-0.1382, 0.9266,  99.115],\
                            [ 0.2979, 0.9123, 114.708]])
        >>> xyz_from_geographic(coords[:,0], coords[:,1], coords[:,2]).T
        array([[3802111.62491437, -528822.82583168, 5076662.15079859],
               [3738960.12012956, 1147998.32536741, 5021398.44437063]])
    """
    wgs84_a = 6378137.0
    wgs84_f = 1.0 / 298.257223563
    wgs84_e2 = wgs84_f * (2.0 - wgs84_f)
    c = normalized_earth_radius(lat_rad)
    f = wgs84_f
    a = wgs84_a
    s = c * (1 - f) ** 2
    return array([(a * c + height_m) * cos(lat_rad) * cos(lon_rad),
                  (a * c + height_m) * cos(lat_rad) * sin(lon_rad),
                  (a * s + height_m) * sin(lat_rad)], dtype=float64)


def normal_vector_ellipsoid(lon_rad, lat_rad):
    """
    Make a vector normal to the ellipsoid at given longitude and latitude

    Examples:
        >>> normal_vector_ellipsoid(0.12, 0.92)
        array([0.60146348, 0.07252407, 0.79560162])
        >>> normal_vector_ellipsoid([[0.12, 0.13]], [[0.92, 0.93]]).shape
        (1, 2, 3)
    """
    normalvector = array([cos(lat_rad) * cos(lon_rad),
                          cos(lat_rad) * sin(lon_rad),
                          sin(lat_rad)])
    return moveaxis(normalvector, 0, -1)


def normal_vector_meridian_plane(xyz_m):
    """
    Return a unit vector normal to the meridian plane.
    If a vector of xyz's is given, xyz should be the fastest varying axis.

    Example:
        >>> test_coord = [3802111.6, -528822.8, 5076662.2]
        >>> normal_vector_meridian_plane(test_coord)
        array([-0.1377605 , -0.99046557,  0.        ])

        >>> normal_vector_meridian_plane(array(test_coord))
        array([-0.1377605 , -0.99046557,  0.        ])

        >>> normal_vector_meridian_plane(array([test_coord, test_coord]))
        array([[-0.1377605 , -0.99046557,  0.        ],
               [-0.1377605 , -0.99046557,  0.        ]])
    """
    result = zeros_like(xyz_m)
    result[..., 0] = array(xyz_m)[..., 1]
    result[..., 1] = -array(xyz_m)[..., 0]
    result[..., 2] = 0
    return result / norm(result, axis=-1, keepdims=True)


def projection_matrix(xyz0_m, normal_vector):
    """
    Create a projection matrix that will project a vector to the plane
    orthogonal to normal_vector, with local north defined at xyz0_m.
    The xyz should be the fastest varying axis.

    Example:
        >>> test_coord = [3802111.6, -528822.8, 5076662.2]
        >>> cs002_normal = [0.59866826, 0.07212702, 0.79774307]
        >>> projection_matrix(test_coord, cs002_normal)
        array([[ 0.04616828, -0.79966543,  0.59866826],
               [ 0.99117465,  0.11122275,  0.07212702],
               [-0.12426302,  0.59005482,  0.79774307]])
        >>> projection_matrix([[test_coord, test_coord]],
        ...                   [[cs002_normal, cs002_normal]]).shape
        (1, 2, 3, 3)
    """
    assert len(xyz0_m) == len(xyz0_m)
    r_unit = array(normal_vector)
    meridian_normal = normal_vector_meridian_plane(xyz0_m)
    q_unit = cross(meridian_normal, r_unit)
    q_unit /= norm(q_unit, axis=-1, keepdims=True)
    p_unit = cross(q_unit, r_unit)
    p_unit /= norm(p_unit, axis=-1, keepdims=True)
    # Swapaxes transposes the 3x3 projection matrices (last two dims)
    return stack([p_unit, q_unit, r_unit], axis=-2).swapaxes(-1, -2)


def transform(xyz_m, xyz0_m, mat):
    """Perform a coordinate transformation on an array of points

    Args:
        xyz_m (array): Array of points
        xyz0_m (array): Origin of transformation
        mat (array): Transformation matrix

    Returns:
        array: Array of transformed points
    """
    offsets = xyz_m - xyz0_m
    return array([dot(mat, offset) for offset in offsets])


LOFAR_XYZ0_m = array([3826574.0, 461045.0, 5064894.5])
LOFAR_REF_MERIDIAN_NORMAL = normal_vector_meridian_plane(LOFAR_XYZ0_m)
