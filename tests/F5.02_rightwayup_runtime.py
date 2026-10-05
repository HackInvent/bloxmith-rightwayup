"""FB1/FB2/FB3/FB4/FB5: real CPU inference in both runtimes and all three package origins."""

import json
from pathlib import Path
import sys
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"tests"), str(Path(__file__).parent)]
from blocs.rightwayup.block import RightwayupBlock
from block_test_packages import install_test_package, prepare_release_run, surface_payload
from rightwayup_fixture import install_cache, picture
from ui_smoke_common import (create_project_api, create_run_api, data_edge, graph_payload, graph_storage_dir,
    http_json, isolated_server, stop_run_api, text_node, wait_for_run_terminal, wait_for_run_predicate)


def main():
    for mode in ("centralized", "zeromq_active"):
        for origin in (None, "managed", "linked"):
            with isolated_server() as server:
                photo = picture(server.root_dir)
                tilted = server.root_dir/"tilted.png"
                with Image.open(photo) as image:
                    image.rotate(-45, expand=True).save(tilted)
                node = RightwayupBlock().build_node_payload(node_id="roll", position={"x": 440, "y": 160},
                    config_overrides={"allowed_root": str(server.root_dir), "download_missing": False, "tier": "nano",
                        "smoothing_window": 1, "min_interval_ms": 0, "max_age_ms": 600000})
                node["outputs"].reverse()
                if origin:
                    model = install_test_package(server, "rightwayup", origin=origin)
                    node["block_version"] = model["version"]
                    for surface in ("modal", "inspector_panel", "node_card"):
                        html = surface_payload(server, model, node, surface)["html"]
                        assert "{{" not in html
                atomic = {"path": str(photo), "source_id": "north", "frame_id": "first"}
                graph = graph_payload("Local camera roll", [text_node("source", "Frame", json.dumps(atomic) if mode == "centralized" else "", 80, 160), node],
                    [data_edge("frame", "source", 1, "roll", 1)])
                project = create_project_api(server, document=graph)["project"]
                storage = graph_storage_dir(server, project)/"instances"/"1"/"blocs"/"roll"
                install_cache(storage, ("nano",))
                def value(data):
                    raw = data.get("output_values", {}).get("roll:1", {}).get("value")
                    return json.loads(raw) if raw else {}
                if mode == "centralized":
                    run = create_run_api(server, graph, project_id=project["graph_id"], runtime_mode=mode)
                    result = wait_for_run_terminal(server, run["run_id"], timeout_sec=30)
                    assert result["status"] == "success", result.get("logs")
                else:
                    rid = prepare_release_run(server, project["graph_id"], graph)["run_id"]
                    assert not list(storage.glob("rightwayup-job-*")), "Run preparation loaded the model"
                    http_json(server.base_url, f"/api/runs/{rid}/play", method="POST", payload={})
                    def send(item):
                        http_json(server.base_url, f"/api/runs/{rid}/active/control", method="POST", payload={
                            "action": "publish_output", "node_id": "source", "port_id": 1,
                            "value": json.dumps(item), "content_type": "application/json"})
                    def wait(ident):
                        data = wait_for_run_predicate(server, rid,
                            lambda data: data.get("status") == "failed" or value(data).get("frame_id") == ident,
                            "RightWayUp did not produce the expected frame", timeout_sec=25)
                        assert data["status"] != "failed", data.get("logs")
                        return data
                    try:
                        send(atomic); result = wait("first")
                        assert value(result)["source_id"] == "north"
                        send({**atomic, "frame_id": "bad-image", "path": str(server.root_dir/"missing.png")})
                        assert value(wait("bad-image"))["state"] == "error"
                        send({**atomic, "frame_id": "after-error", "source_id": "south"})
                        result = wait("after-error")
                        assert value(result)["source_id"] == "south" and value(result)["state"] != "error"
                        send({**atomic, "frame_id": "rotation", "path": str(tilted)})
                        changed = wait("rotation")
                        alert = json.loads(changed["output_values"]["roll:2"]["value"])
                        assert alert["event"] == "rotated" and alert["frame_id"] == "rotation", alert
                        assert 35 < alert["delta_cw"] < 55 and alert["source_id"] == "north", alert
                        # Fresh frames after an error are processed, not permanently poisoned.
                        for index in range(10):
                            send({**atomic, "frame_id": "burst-"+str(index)})
                        result = wait("burst-9")
                        returned = json.loads(result["output_values"]["roll:2"]["value"])
                        assert returned["event"] == "returned" and returned["source_id"] == "north", returned
                    finally:
                        started = time.monotonic()
                        stop_run_api(server, rid)
                        wait_for_run_terminal(server, rid, timeout_sec=15)
                        assert time.monotonic()-started < 5
                orientation = value(result)
                assert orientation["model"]["name"] == "RightWayUp", result.get("logs")
                assert orientation["model"]["tier"] == "nano"
                assert 0 <= orientation["confidence"] <= 1
                assert orientation["source"]["bytes"] == photo.stat().st_size
                assert orientation["state"] == "stable" and orientation["accepted"] is True
                cleanup_deadline = time.monotonic()+2
                while list(storage.glob("rightwayup-job-*")) and time.monotonic() < cleanup_deadline:
                    time.sleep(.02)
                leftovers = {path.name: [part.name for part in path.iterdir()] for path in storage.glob("rightwayup-job-*")}
                assert not leftovers, (mode, origin, leftovers, result.get("logs"))
            print("[ok] RightWayUp real model runtime " + mode + " " + str(origin), flush=True)


if __name__ == "__main__":
    main()
