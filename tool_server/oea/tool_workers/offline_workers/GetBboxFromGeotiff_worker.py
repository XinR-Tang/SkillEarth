import os
import argparse
import rasterio
from rasterio.warp import transform_bounds

from tool_server.oea.utils.utils import *
from tool_server.oea.utils.server_utils import *

logger = build_logger("geotiff_bbox_worker")


def GetBboxFromGeotiff(geotiff, dst_crs_authid="EPSG:4326"):
    """
    Extract bbox (west, south, east, north) from a GeoTIFF file, in dst_crs_authid.
    """
    with rasterio.open(geotiff) as src:
        logger.info(f"Opened raster: {geotiff}")
        logger.info(f"Source CRS: {src.crs}")
        logger.info(f"Source bounds: {src.bounds}")

        if src.crs is None:
            raise ValueError(f"Input GeoTIFF has no CRS: {geotiff}")

        b = src.bounds

        if src.crs.to_string() == dst_crs_authid:
            west, south, east, north = b.left, b.bottom, b.right, b.top
        else:
            west, south, east, north = transform_bounds(
                src.crs,
                dst_crs_authid,
                b.left,
                b.bottom,
                b.right,
                b.top,
                densify_pts=21,
            )

    bbox = tuple(round(v, 4) for v in (west, south, east, north))
    logger.info(f"Extracted Bbox (W, S, E, N): {bbox}")
    return bbox


def generate(params):
    if "geotiff" not in params:
        txt_e = "Missing required parameter: geotiff"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 2}

    geotiff_path = params.get("geotiff")
    dst_crs = params.get("dst_crs", "EPSG:4326")

    if not os.path.isfile(geotiff_path):
        txt_e = f"GeoTIFF not found: {geotiff_path}"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 3}

    try:
        bbox = GetBboxFromGeotiff(geotiff_path, dst_crs)
        txt = f"Bbox (west, south, east, north): {bbox}"
        return {"text": txt, "error_code": 0}

    except Exception as e:
        txt_e = f"Error extracting bbox from GeoTIFF: {repr(e)}"
        logger.exception(txt_e)
        return {"text": txt_e, "error_code": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geotiff", type=str, required=True, help="Path to GeoTIFF file.")
    parser.add_argument("--dst-crs", type=str, default="EPSG:4326")
    args = parser.parse_args()

    params = {"geotiff": args.geotiff, "dst_crs": args.dst_crs}
    out = generate(params)
    print(out)