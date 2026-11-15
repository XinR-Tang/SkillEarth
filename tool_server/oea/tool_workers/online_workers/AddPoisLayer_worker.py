import argparse
import multiprocessing
import uuid
import os
import socket
import pandas as pd
import geopandas as gpd
import osmnx as ox
import requests
from datetime import datetime

from tool_server.oea.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.oea.utils.server_utils import build_logger

CRS = "EPSG:4326"

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"AddPoisLayer_worker_{worker_id}.log")


def configure_osmnx() -> None:
    """Use the launch proxy and a persistent local cache for OSM requests."""
    cache_dir = os.environ.get(
        "OEA_OSMNX_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), "..", "cache", "osmnx"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    # Keep each endpoint attempt below the complete task deadline so the
    # fallback endpoint still gets a chance within the 10-second budget.
    ox.settings.requests_timeout = int(os.environ.get("OEA_OVERPASS_REQUEST_TIMEOUT", "9"))

    mac_proxy = os.environ.get("OEA_MAC_NETWORK_PROXY", "").strip()
    if not mac_proxy:
        try:
            with socket.create_connection(("127.0.0.1", 5620), timeout=0.5):
                mac_proxy = "socks5h://127.0.0.1:5620"
        except OSError:
            pass
    proxy = mac_proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http_proxy = mac_proxy or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or proxy
    if proxy or http_proxy:
        kwargs = dict(getattr(ox.settings, "requests_kwargs", {}) or {})
        kwargs["proxies"] = {"http": http_proxy or proxy, "https": proxy or http_proxy}
        ox.settings.requests_kwargs = kwargs


def overpass_endpoints():
    """Return ordered Overpass endpoints with an environment override."""
    configured = os.environ.get("OEA_OVERPASS_URLS", "")
    values = [value.strip().rstrip("/") for value in configured.split(",") if value.strip()]
    if values:
        return values
    # Use the endpoint currently reachable through the Mac reverse SOCKS
    # tunnel. overpass-api.de is intentionally omitted: when it is down, every
    # attempt emits an SSH connect_to error and consumes half the task budget.
    return [
        "https://overpass.private.coffee/api",
    ]


def clean_for_gpkg(gdf):
    gdf = gdf.copy()
    gdf.columns = gdf.columns.str.lower().str[:30]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf

def get_name_row(row):
    """Return best display name for OSM features in GeoDataFrame (no pandas)."""
    fields = row.index
    # Priority: name:en > name
    for key in ("name:en", "name"):
        if key in fields:
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    # Any other name:* field
    for key in fields:
        if key.startswith("name"):
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    return ""

def add_pois_layer(gpkg, query, layer_name):
    """
    Add POIs into GeoPackage from OSM
    Ensures features fall inside boundary polygon stored in layer "area_boundary".

    Parameters
    ----------
    gpkg : str
        Path to GeoPackage
    query : dict or str
        - dict: tags (e.g., {"amenity": "hospital"})
        - str: single POI name (e.g., "Charité Hospital, Berlin")
    layer_name : str
        New layer name to save in GeoPackage
    
    Returns
    -------
    str
        Name of the created layer.
    """
    configure_osmnx()
    # Load boundary
    area_gdf = gpd.read_file(gpkg, layer="area_boundary")
    geom = area_gdf.geometry.iloc[0]

    if isinstance(query, dict):  # tags
        if len(query) != 1:
            raise ValueError(f"Query must contain exactly one key, got: {list(query.keys())}")
        key, value = next(iter(query.items()))
        if not (
            isinstance(value, (str, bool)) or
            (isinstance(value, list) and all(isinstance(v, str) for v in value))
        ):
            raise ValueError(
                f"Invalid query value for key '{key}': "
                f"must be str, bool, or list[str], got {type(value).__name__}"
            )

        network_errors = []
        for endpoint in overpass_endpoints():
            ox.settings.overpass_url = endpoint
            try:
                pois = ox.features_from_polygon(geom, query).to_crs(CRS)
                break
            except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout, requests.exceptions.SSLError) as exc:
                network_errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
                logger.warning("Overpass endpoint failed: %s", network_errors[-1])
        else:
            raise requests.exceptions.ConnectionError(
                "All configured Overpass endpoints failed. " + " | ".join(network_errors)
            )

    elif isinstance(query, str):  # name
        pois = ox.geocode_to_gdf(query).to_crs(CRS)
        pois = pois[pois.geometry.within(geom)]
        if pois.empty:
            raise ValueError(f"POI '{query}' not found inside study area.")

    else:
        raise ValueError("query must be dict or str for OSM")
    
    pois["display_name"] = pois.apply(get_name_row, axis=1)
    pois = pois[pois["display_name"] != ""]

    if pois.empty:
        raise ValueError(f"No POIs found for {query} inside boundary.")
    
    # Deduplicate by name → keep first occurrence
    pois = pois.drop_duplicates(subset="display_name", keep="first")
    
    # Clean and save
    pois = clean_for_gpkg(pois)
    pois.to_file(gpkg, layer=layer_name, driver="GPKG")
    return layer_name, len(pois)


def _add_pois_process(send_connection, gpkg, query, layer_name):
    """Run one OSM request in an isolated process so it can be terminated."""
    try:
        layer, count = add_pois_layer(gpkg, query, layer_name)
        send_connection.send(("ok", layer, count))
    except BaseException as exc:
        send_connection.send((
            "error", type(exc).__module__, type(exc).__name__, str(exc),
        ))
    finally:
        send_connection.close()


def add_pois_layer_with_timeout(gpkg, query, layer_name, timeout):
    """Execute a complete POI request with a hard wall-clock timeout."""
    # This worker runs on Linux and has already imported the geospatial stack.
    # Fork inherits those loaded modules. Using spawn here re-imports OSMnx and
    # NetworkX for every asset, which is both slow and unreliable on the shared
    # site-packages filesystem used by the tool-server environment.
    context = multiprocessing.get_context("fork")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_add_pois_process,
        args=(send_connection, gpkg, query, layer_name),
        daemon=False,
    )
    process.start()
    send_connection.close()
    try:
        if not receive_connection.poll(timeout):
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            raise TimeoutError(
                f"Overpass request exceeded the {timeout:.0f}s task timeout"
            )
        try:
            result = receive_connection.recv()
        except EOFError as exc:
            process.join(5)
            raise RuntimeError(
                "POI subprocess exited without returning a result "
                f"(exitcode={process.exitcode})"
            ) from exc
    finally:
        receive_connection.close()
        if process.is_alive():
            process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)

    if result[0] == "ok":
        return result[1], result[2]
    _, module_name, exception_name, message = result
    if (
        module_name.startswith(("requests", "urllib3"))
        or any(token in exception_name for token in ("SSL", "Timeout", "Connection", "Proxy"))
    ):
        raise requests.exceptions.ConnectionError(
            f"{exception_name}: {message}"
        )
    raise RuntimeError(f"{exception_name}: {message}")

class AddPoisLayerWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="AddPoisLayer",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=10.0,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info("AddPoisLayerWorker does not need a model. Ready to run.")

    def generate(self, params):
        required_keys = ("gpkg", "query", "layer_name")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        query = params.get("query")
        layer_name = params.get("layer_name")
        
        if not os.path.exists(gpkg):
            txt_e = f"GeoPackage not found"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}
        
        try:
            layer, count = add_pois_layer_with_timeout(
                gpkg, query, layer_name, self.task_timeout,
            )
            gpkg_name = os.path.basename(gpkg)
            txt = f"Saved {count} POIs to layer '{layer}' in {gpkg_name}"
            return {"text": txt, "error_code": 0}

        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, TimeoutError) as e:
            txt_e = (
                "Error in AddPoisLayer: OSM network request failed; keep the query unchanged "
                f"and retry after checking the launch proxy. {type(e).__name__}: {e}"
            )
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 4}

        except Exception as e:
            txt_e = f"Error in AddPoisLayer: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}


    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "AddPoisLayer",
                "description": "Adds Points of Interest (POIs) into a GeoPackage inside the area boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "query": {"type": "object", "description": "OSM query as dict or POI name as string."},
                        "layer_name": {"type": "string", "description": "Layer name for saving POIs."}
                    },
                    "required": ["gpkg", "query", "layer_name"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20018)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="AddPoisLayer")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--task-timeout", type=float, default=10.0)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = AddPoisLayerWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        task_timeout=args.task_timeout,
        host=args.host,
        port=args.port,
    )
    worker.run()
