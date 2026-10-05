"""Local orientation and bounded per-source tracking using public block services only."""

import json
from bloxsmith_app.block_api import BlockDefinition, BlockRuntimeOutput, BlockRuntimePreparation, BlockRuntimeResult
from . import engine, logic, ui


# FB1 - Validate fresh local frames, correlation and bounded model settings; no cached replay.
# FB2 - Load checksum-pinned ORTUS ONNX weights and execute real local CPU/GPU inference.
# FB3 - Circular smoothing, confidence abstention, baseline drift and source/time isolation.
# FB4 - Latest pending frame per source, bounded scheduling and cancellable owned processes.
# FB5 - Public runtime/distribution contracts and responsive translated draft settings.
class RightwayupBlock(BlockDefinition):
    kind = "rightwayup"

    def ui_assets(self, surface="modal"):
        return list(self.model.get("ui_assets", {}).get(surface, []))

    def prepare_runtime(self, context):
        logic.configuration(context.config)
        return BlockRuntimePreparation(listen_on_run=context.runtime_mode == "zeromq_active")

    def result(self, context, values):
        outputs = [BlockRuntimeOutput(port_id=int(port.id), port_name=port.name,
            value=json.dumps(values[port.name], ensure_ascii=True, allow_nan=False), content_type="application/json")
            for port in context.output_ports if port.name in values]
        return BlockRuntimeResult(status="success", outputs=outputs)

    def failure(self, context, exc, item=None):
        item = item or {}
        return self.result(context, {"orientation": {"source_id": item.get("source_id"), "frame_id": item.get("frame_id"),
            "timestamp": item.get("timestamp"), "accepted": False, "angle_cw": None,
            "state": "cancelled" if isinstance(exc, logic.Cancelled) else "error",
            "detail": str(exc) if isinstance(exc, logic.OrientationError) else "Local orientation is unavailable."}})

    def storage(self, context):
        resolver = context.services.get("get_block_storage_dir")
        if not callable(resolver):
            raise logic.OrientationError("The block storage service is unavailable.")
        return resolver()

    def execute_runtime(self, context):
        session, item = None, None
        try:
            cfg = logic.configuration(context.config)
            by_name = {port.name: port.id for port in context.input_ports}
            if context.input_events:
                values = [event.value for event in context.input_events if event.input_port_id == by_name.get("frame")]
            else:
                attribute = context.input_attribute("frame", "1")
                values = [attribute.value] if attribute and attribute.status == "updated" else []
            if not values:
                return BlockRuntimeResult(status="skipped", outputs=[])
            if len(values) != 1:
                raise logic.OrientationError("Send one frame per event, not a batch.")
            item = logic.frame(values[0], cfg)
            if context.runtime_mode == "zeromq_active":
                listener = context.services.get("runtime_listener")
                if listener is None:
                    raise logic.OrientationError("The runtime listener bridge is unavailable.")
                listener.send(item)
                return BlockRuntimeResult(status="success", outputs=[])
            session = engine.Session(cfg, context.root_dir, self.storage(context))
            prediction = session.predict(item, context.services.get("cancel_requested", lambda: False))
            return self.result(context, logic.Tracker(cfg).apply(item, prediction))
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            return self.failure(context, exc, item)
        finally:
            if session:
                session.close()

    def listen_runtime(self, context):
        cfg = logic.configuration(context.config)
        def emit(values):
            if not context.stop_requested():
                context.emit_result(self.result(context, values))
        listener = None
        try:
            listener = engine.Listener(cfg, context.root_dir, self.storage(context), emit)
            while not context.stop_requested():
                received = context.receive_command(timeout_sec=.02)
                if received is not None:
                    try:
                        listener.offer(logic.frame(dict(received.payload), cfg))
                    except (ValueError, TypeError):
                        emit({"orientation": {"accepted": False, "state": "error", "angle_cw": None,
                            "detail": "Invalid frame message."}})
                listener.tick()
        except (OSError, RuntimeError, ValueError) as exc:
            if not context.stop_requested():
                context.emit_result(self.failure(context, exc))
            while not context.stop_requested():
                context.receive_command(timeout_sec=.1)
        finally:
            if listener:
                listener.close()

    def handle_ui_action(self, *, node, action, values, payload=None):
        if action in {"save_settings", "modal_update_fields", "inspector_update_fields"}:
            patch = (values or {}).get("node_patch") or {}
            if "config" in patch:
                try:
                    logic.configuration({**self.default_config(), **(node.get("config") or {}), **patch["config"]})
                except (ValueError, TypeError) as exc:
                    return {"error": self.translate("block.rightwayup.error", {"detail": str(exc)}, fallback="Invalid settings: {detail}")}
        return ui.replace_config(super().handle_ui_action(node=node,
            action="modal_update_fields" if action == "save_settings" else action, values=values, payload=payload), node)

    def render_modal(self, *, node, payload=None):
        return ui.modal(self, node, payload)

    def render_inspector_panel(self, *, node, payload=None):
        return ui.inspector(self, node, payload)

    def render_node_card(self, *, node, payload=None):
        cfg = {**self.default_config(), **(node.get("config") or {})}
        return ui.card(self, node, f"ORTUS · {cfg['tier']} · {cfg['device'].upper()}")
