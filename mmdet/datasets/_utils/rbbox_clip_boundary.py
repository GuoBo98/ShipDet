import pdb
# import geopandas
from shapely.geometry import Polygon
import numpy as np


def polygon2mask(polygon):
    """convet polygon to mask
    Arguments:
        polygon {Polygon} -- input polygon (single polygon)
    Returns:
        list -- converted mask ([x1, y1, x2, y2, ...])
    """
    mask = np.array(polygon.exterior.coords, dtype=int)[:-1].ravel().tolist()
    return mask


def clip_boundary_polygon(polygons, image_size=(1024, 1024)):
    h, w = image_size
    image_boundary = Polygon([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1), (0, 0)])

    # polygons = geopandas.GeoDataFrame({'geometry': polygons, 'polygon_df': range(len(polygons))})

    # clipped_polygons = geopandas.clip(polygons, image_boundary).to_dict()
    # clipped_polygons = list(clipped_polygons['geometry'].values())
    # return clipped_polygons
    return polygons

def find_max_area_5point(poly):
    """
    :param poly: 5 point poly(list)
    :return poly: 4 point poly with max area
    """
    assert len(poly) >= 10 and len(poly) % 2 == 0
    num_coord = int(len(poly) / 2)
    poly = np.array(poly)
    combinate = [np.roll(poly, i * 2)[:8] for i in range(num_coord)]
    # if num_coord > 5:
    #     import pdb
    #     pdb.set_trace()
    #     print(combinate)
    # combinate = [np.roll(poly, 0)[:8], np.roll(poly, 2)[:8], np.roll(poly, 4)[:8], np.roll(poly, 6)[:8], np.roll(poly, 8)[:8]]
    area = np.array([Polygon(poly_sub.reshape(-1, 2)).area for poly_sub in combinate])
    max_ind = np.argmax(area)
    return combinate[max_ind].tolist()


# if __name__ == '__main__':
#     tmp_a = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
#     print((tmp_a.shape))
#     poly = [Polygon(tmp_a)]
#     print(poly[0].area)
#     print((clip_boundary_polygon(poly)[0]))
#     # print(polygon2mask(clip_boundary_polygon(poly)[0]))
