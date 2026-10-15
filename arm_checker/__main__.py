"""Launch the local checker; no simulator, GPU or model weights are needed."""
import argparse
import uvicorn
from .app import create_app


def main():
    parser = argparse.ArgumentParser(description="PhoneToPanda arm checker")
    parser.add_argument("--demo", action="store_true", help="Use labelled synthetic sensor motion")
    parser.add_argument("--port", default="auto", help="Serial device, or auto when one compatible device is connected")
    parser.add_argument("--protocol", choices=("am", "json"), default="am", help="Existing ESP32 AM v1 packets or JSON lines")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--sessions-dir", default=None)
    args = parser.parse_args()
    app = create_app(source="demo" if args.demo else "serial", port=args.port,
                     protocol=args.protocol, baud=args.baud, sessions_dir=args.sessions_dir)
    print(f"Arm checker: http://127.0.0.1:{args.http_port} (local computer only)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.http_port, log_level="info")


if __name__ == "__main__":
    main()
