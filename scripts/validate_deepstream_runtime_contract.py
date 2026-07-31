from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from pyservicemaker import (
    BatchMetadataOperator,
    BufferOperator,
    Pipeline,
    Probe,
    osd,
)

from rtsp_annotator.deepstream_worker import (
    OverlayProcessor,
    StreamPolicy,
    _add_pipeline_nodes,
)


def validate_metadata_style() -> None:
    rectangle = SimpleNamespace(
        left=10.0,
        top=30.0,
        width=100.0,
        height=200.0,
        border_width=0,
        border_color=None,
    )
    text = osd.TextParams()
    object_meta = SimpleNamespace(
        class_id=0,
        confidence=0.9,
        rect_params=rectangle,
        text_params=text,
    )
    policy = StreamPolicy(
        stream_id="validation",
        classes=frozenset({0}),
        conf=0.25,
        roi=None,
        labels={0: "人员"},
    )

    OverlayProcessor._style_object(object_meta, policy, osd)

    assert text.display_text == "人员"
    assert text.font_params.size == 18
    assert text.set_bg_clr


def validate_request_pad_links() -> None:
    demux_pipeline = Pipeline("validate-demux-request-pads")
    demux_pipeline.add("nvstreamdemux", "demux")
    demux_pipeline.add("queue", "queue_0")
    demux_pipeline.add("queue", "queue_1")
    demux_pipeline.link(
        ("demux", "queue_0"),
        ("src_%u", ""),
    )
    demux_pipeline.link(
        ("demux", "queue_1"),
        ("src_%u", ""),
    )

    sink_pipeline = Pipeline("validate-rtsp-request-pad")
    sink_pipeline.add("h264parse", "parser")
    sink_pipeline.add("rtspclientsink", "sink")
    sink_pipeline.link(
        ("parser", "sink"),
        ("", "sink_%u"),
    )


def validate_encoded_buffer_probe() -> None:
    class Counter(BufferOperator):
        def handle_buffer(self, _buffer: object) -> bool:
            return True

    pipeline = Pipeline("validate-encoded-buffer-probe")
    pipeline.add("h264parse", "parser")
    pipeline.add(
        "capsfilter",
        "au_caps",
        {
            "caps": (
                "video/x-h264, "
                "stream-format=byte-stream, alignment=au"
            )
        },
    )
    pipeline.add(
        "clocksync",
        "clock",
        {"sync": True, "sync-to-first": True},
    )
    pipeline.add("fakesink", "sink")
    pipeline.link("parser", "au_caps", "clock", "sink")
    pipeline.attach("clock", Probe("encoded-counter", Counter()))


def validate_complete_pipeline_construction() -> None:
    class MetadataPass(BatchMetadataOperator):
        def handle_metadata(self, _batch_meta: object) -> None:
            pass

    class BufferPass(BufferOperator):
        def handle_buffer(self, _buffer: object) -> bool:
            return True

    pipeline = Pipeline("validate-complete-pipeline")
    latency_probe = Probe("latency", MetadataPass())
    overlay_probe = Probe("overlay", MetadataPass())

    def buffer_probe(index: int) -> Probe:
        return Probe(f"buffer-{index}", BufferPass())

    _add_pipeline_nodes(
        pipeline,
        {
            "gpu_id": 0,
            "batch_size": 1,
            "batch_push_timeout_us": 20_000,
            "mux_width": 1920,
            "mux_height": 1080,
            "source_latency_ms": 100,
            "encoder_iframe_interval": 25,
            "tracker_config": (
                "/opt/nvidia/deepstream/deepstream/samples/configs/"
                "deepstream-app/config_tracker_NvDCF_perf.yml"
            ),
            "tracker_library": (
                "/opt/nvidia/deepstream/deepstream/lib/"
                "libnvds_nvmultiobjecttracker.so"
            ),
            "streams": [
                {
                    "input_url": "rtsp://127.0.0.1/input",
                    "output_url": "rtsp://127.0.0.1/output",
                    "bitrate_bps": 2_500_000,
                }
            ],
        },
        Path(
            "/opt/nvidia/deepstream/deepstream/samples/configs/"
            "deepstream-app/config_infer_primary.txt"
        ),
        latency_probe,
        overlay_probe,
        overlay_probe,
        buffer_probe,
        buffer_probe,
    )


def main() -> None:
    validate_metadata_style()
    validate_request_pad_links()
    validate_encoded_buffer_probe()
    # Docker builds intentionally have no GPU attached. The complete
    # DeepStream graph can be requested explicitly on a GPU host, while the
    # build always validates the request pads, clock element and probe API.
    if os.environ.get("VALIDATE_COMPLETE_PIPELINE") == "1":
        validate_complete_pipeline_construction()
    print("DEEPSTREAM_RUNTIME_CONTRACT_OK")


if __name__ == "__main__":
    main()
