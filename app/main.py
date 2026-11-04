"""
SUTRA Vision Engine Main Entrypoint.
Provides command-line interfaces for live webcam execution, video replay,
synthetic protocol simulations, deviation verification, and benchmarking.
"""

import argparse
import logging
import sys
from pathlib import Path
import time
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SutraConfig, get_config
from app.engine import SutraVisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("sutra.main")


def run_pipeline_loop(engine: SutraVisionEngine, max_frames: int = 0, headless: bool = False):
    """Execution loop consuming frames from the zero-lag frame buffer."""
    engine.start()
    processed = 0

    try:
        while True:
            ingested = engine.frame_buffer.get_latest(timeout=1.0)
            if ingested is None:
                if not engine.ingestor.is_connected:
                    logger.info("Waiting for video source...")
                    time.sleep(0.5)
                    continue
                else:
                    logger.info("End of video stream reached.")
                    break

            res = engine.process_single_frame(ingested, annotate=not headless)
            processed += 1

            if processed % 30 == 0:
                diag = engine.get_diagnostics()
                logger.info(
                    f"Frame {res.frame_id:04d} | Step: {res.current_step_id} | "
                    f"Lat: {res.latency_ms:.1f}ms (P95: {diag['latency_p95_ms']:.1f}ms) | "
                    f"RAM: {diag['memory_rss_mb']:.1f}MB | Drops: {diag['buffer']['total_dropped']}"
                )

            # Optional display
            if not headless and res.annotated_frame is not None:
                try:
                    cv2.imshow("SUTRA - AI On-board BAS Experiment Guardian", res.annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        logger.info("User requested exit.")
                        break
                    elif key == ord('a'):  # Operator acknowledgement hotkey
                        engine.protocol_engine.acknowledge_deviation()
                except (cv2.error, Exception):
                    pass

            if max_frames > 0 and processed >= max_frames:
                logger.info(f"Reached max frame count of {max_frames}.")
                break

    except KeyboardInterrupt:
        logger.info("Shutdown signaled by user.")
    finally:
        engine.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        diag = engine.get_diagnostics()
        print("\n" + "=" * 60)
        print("          SUTRA VISION ENGINE RUN DIAGNOSTICS")
        print("=" * 60)
        print(f"Total Frames Processed : {diag['processed_frames']}")
        print(f"Mean Latency           : {diag['latency_mean_ms']} ms")
        print(f"P95 Latency            : {diag['latency_p95_ms']} ms")
        print(f"Peak Memory (RSS)      : {diag['memory_rss_mb']} MB")
        print(f"Frames Ingested        : {diag['buffer']['total_ingested']}")
        print(f"Frames Dropped (Lag-0) : {diag['buffer']['total_dropped']} ({diag['buffer']['drop_rate_pct']}%)")
        print(f"Telemetry Events Logged: {diag['telemetry']['total_events']} ({diag['telemetry']['size_kb']} KB)")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SUTRA: AI Human Activity Recognition for On-board BAS Experiments")
    parser.add_argument("--mode", choices=["live", "video", "synthetic", "benchmark", "test-deviation"], default="live",
                        help="Execution mode")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: camera index (0) or path to video file")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window")
    parser.add_argument("--frames", type=int, default=0,
                        help="Number of frames to process before stopping (0 = infinite)")
    parser.add_argument("--publish", action="store_true",
                        help="Enable HTTP telemetry publisher to mock receiver")
    parser.add_argument("--web", action="store_true",
                        help="Launch interactive browser-based real-time video HUD (http://127.0.0.1:8081)")
    parser.add_argument("--port", type=int, default=8081,
                        help="Port for browser web HUD (default: 8081)")
    args = parser.parse_args()

    config = get_config()

    if args.source:
        config.video_source = args.source
    if args.publish:
        config.enable_http_publisher = True
        config.publisher_url = config.publisher_url or "http://127.0.0.1:8088/api/v1/telemetry/events"

    if args.mode == "benchmark":
        from benchmarks.benchmark_suite import run_benchmark
        run_benchmark()
        return

    # Web HUD Mode: opens real-time browser stream at http://127.0.0.1:{args.port}
    if args.web:
        from app.web_viewer import start_web_server
        if args.mode == "live" and not args.source:
            config.video_source = "0"
        engine = SutraVisionEngine(config)
        if args.mode == "synthetic":
            from scripts.generate_synthetic_video import create_synthetic_frame_generator
            engine.ingestor.source = create_synthetic_frame_generator(simulate_deviation=False)
            engine.ingestor._source_type = "synthetic"
        elif args.mode == "test-deviation":
            from scripts.generate_synthetic_video import create_synthetic_frame_generator
            engine.ingestor.source = create_synthetic_frame_generator(simulate_deviation=True)
            engine.ingestor._source_type = "synthetic"
        start_web_server(engine, host="127.0.0.1", port=args.port)
        return

    if args.mode == "synthetic":
        from scripts.generate_synthetic_video import create_synthetic_frame_generator
        gen = create_synthetic_frame_generator(simulate_deviation=False)
        engine = SutraVisionEngine(config)
        engine.ingestor.source = gen
        engine.ingestor._source_type = "synthetic"
        run_pipeline_loop(engine, max_frames=args.frames or 180, headless=args.headless)
        return

    if args.mode == "test-deviation":
        from scripts.generate_synthetic_video import create_synthetic_frame_generator
        gen = create_synthetic_frame_generator(simulate_deviation=True)
        engine = SutraVisionEngine(config)
        engine.ingestor.source = gen
        engine.ingestor._source_type = "synthetic"
        run_pipeline_loop(engine, max_frames=args.frames or 180, headless=args.headless)
        return

    # Standard live or video file mode
    if args.mode == "live" and not args.source:
        config.video_source = "0"

    engine = SutraVisionEngine(config)
    run_pipeline_loop(engine, max_frames=args.frames, headless=args.headless)


if __name__ == "__main__":
    main()
