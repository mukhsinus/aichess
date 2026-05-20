"""
Camera diagnostic utility.
1. Lists video devices that Windows sees (via PowerShell / PnP).
2. Probes indexes 0-9 with three OpenCV backends: auto, MSMF, DirectShow.
3. Shows live feed from any working camera.
Controls (in the VIDEO window): N = next, P = prev, Q = quit
"""

import subprocess
import sys

import cv2

MAX_INDEX = 9

BACKENDS = [
    ("auto",       cv2.CAP_ANY),
    ("MSMF",       cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
]


def list_windows_devices():
    print("=" * 60)
    print("WINDOWS VIDEO CAPTURE DEVICES (PnP)")
    print("=" * 60)
    try:
        ps = (
            "Get-PnpDevice -Class Camera -Status OK "
            "| Format-Table -Property Status, Class, FriendlyName, InstanceId -AutoSize"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        )
        out = result.stdout.strip()
        if out:
            print(out)
        else:
            print("  (no devices in 'Camera' class)")
    except Exception as e:
        print(f"  PnP query failed: {e}")

    try:
        ps2 = (
            "Get-PnpDevice -Class 'Image' -Status OK "
            "| Format-Table -Property Status, Class, FriendlyName, InstanceId -AutoSize"
        )
        result2 = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps2],
            capture_output=True, text=True, timeout=10,
        )
        out2 = result2.stdout.strip()
        if out2:
            print("\nDevices in 'Image' class:")
            print(out2)
    except Exception:
        pass

    print()


def probe_all():
    """Return dict  {(backend_name, index): True}  for working combos."""
    found = {}
    print("=" * 60)
    print("OPENCV BACKEND / INDEX PROBE")
    print("=" * 60)

    for bname, bflag in BACKENDS:
        print(f"\n--- backend: {bname} ---")
        for idx in range(MAX_INDEX + 1):
            label = f"  [{bname}] index {idx}"
            print(f"{label:35s}", end="", flush=True)
            try:
                cap = cv2.VideoCapture(idx, bflag)
                if cap.isOpened():
                    ok, _ = cap.read()
                    cap.release()
                    if ok:
                        print("  OK  <<<")
                        found[(bname, idx)] = True
                        continue
                    else:
                        print("  opened but no frame")
                        continue
                cap.release()
                print("  --")
            except Exception as e:
                print(f"  error: {e}")

    print()
    return found


def show_camera(backend_flag, index):
    cap = cv2.VideoCapture(index, backend_flag)
    if not cap.isOpened():
        print(f"[!] Could not open index {index}")
        return "fail"

    bname = next((n for n, f in BACKENDS if f == backend_flag), "?")
    title = f"Camera {index} ({bname})"
    print(f"[*] Showing {title}")
    print("    >>> Press keys in the VIDEO WINDOW:  N=next  P=prev  Q=quit <<<")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[!] {title}: frame grab failed")
            cap.release()
            return "fail"

        cv2.putText(frame, title, (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "N=next  P=prev  Q=quit", (20, 95),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            return "quit"
        if key == ord("n"):
            cap.release()
            return "next"
        if key == ord("p"):
            cap.release()
            return "prev"


def main():
    list_windows_devices()
    found = probe_all()

    if not found:
        print("No working cameras found with any backend.")
        sys.exit(1)

    entries = []
    for (bname, idx) in sorted(found.keys(), key=lambda x: (x[1], x[0])):
        bflag = next(f for n, f in BACKENDS if n == bname)
        entries.append((bname, bflag, idx))

    print("=" * 60)
    print("WORKING CAMERAS")
    print("=" * 60)
    for i, (bname, _, idx) in enumerate(entries):
        print(f"  {i}: index {idx}  (backend: {bname})")
    print()

    pos = 0
    while True:
        bname, bflag, idx = entries[pos]
        result = show_camera(bflag, idx)

        if result == "quit":
            break
        elif result == "next":
            pos = (pos + 1) % len(entries)
        elif result == "prev":
            pos = (pos - 1) % len(entries)
        elif result == "fail":
            pos = (pos + 1) % len(entries)

    cv2.destroyAllWindows()

    print(f"\nWorking cameras:")
    for i, (bname, _, idx) in enumerate(entries):
        print(f"  {i}: index {idx}  (backend: {bname})")


if __name__ == "__main__":
    main()
