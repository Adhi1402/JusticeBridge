"""
Multi-device transport — AI PC → UNO Q (arch doc Section 5).

The Output agent produces `signal_packet`; this posts it to the FastAPI/MQTT
listener on the UNO Q Debian side, which then drives the STM32 LED matrix +
OLED via the Arduino RPC bridge. Kept deliberately tiny and dependency-light
(stdlib only) so it runs anywhere.

Graceful degradation (a scored "reliability" point): if the UNO Q is
unreachable, sending fails softly and returns False — the phone still speaks
the answer, so the session is never blocked on the kiosk hardware.

    from justicebridge.signal_client import send_signal
    send_signal(state["signal_packet"])            # -> True/False
"""

import json
import urllib.request
import urllib.error

DEFAULT_UNO_Q_URL = "http://uno-q.local:8000/signal"


def send_signal(signal_packet: dict, url: str = DEFAULT_UNO_Q_URL, timeout: float = 3.0) -> bool:
    """POST the signal packet to the UNO Q. Returns True on success, False if
    the device is unreachable (never raises — degradation is a feature)."""
    data = json.dumps(signal_packet).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[signal_client] UNO Q unreachable ({e}); phone-only fallback.")
        return False


if __name__ == "__main__":
    demo = {
        "severity": "amber", "category": "wages", "confidence": 0.62,
        "deadline_days": 90,
        "dlsa": {"name": "DLSA Kanchipuram", "phone": "1516",
                 "bring": "ID + work proof"},
        "qualifies_for_aid": True,
    }
    print("sent:", send_signal(demo))
