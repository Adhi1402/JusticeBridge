"""
Mock UNO Q listener — stands in for the FastAPI/MQTT endpoint on the UNO Q
Debian side so the AI PC → UNO Q contract can be tested WITHOUT the board.

It receives a signal packet and prints what the STM32 tier would render:
the LED colour + the OLED lines (category, deadline, DLSA). On the real board
this handler forwards to the STM32U585 over the Arduino RPC bridge to light
the 8x13 matrix and draw the OLED.

Run (needs FastAPI + uvicorn on the demo/UNO-Q machine):
    pip install fastapi uvicorn
    python -m justicebridge.uno_q_listener
Then, from the AI PC:
    python -m justicebridge.signal_client   # points at DEFAULT_UNO_Q_URL

Graceful-degradation note (arch doc): even standalone, this side owns the
offline DLSA directory, so if the AI PC is down it can still show category +
nearest aid office.
"""

LED = {"red": "🔴 RED  (act now)", "amber": "🟠 AMBER (act soon)", "green": "🟢 GREEN (awareness)"}


def render(packet: dict) -> str:
    sev = packet.get("severity", "green")
    dl = packet.get("deadline_days")
    dl_txt = f"~{max(1, round(dl/7))} wks" if dl else "no deadline"
    dlsa = packet.get("dlsa", {})
    aid = "FREE legal aid: you likely qualify" if packet.get("qualifies_for_aid") else "Ask about free legal aid"
    lines = [
        "┌────────────── UNO Q ──────────────┐",
        f"  LED    : {LED.get(sev, sev)}",
        f"  OLED L1: {str(packet.get('category','')).upper()} · act within {dl_txt}",
        f"  OLED L2: {aid}",
        f"  OLED L3: {dlsa.get('name','')} · {dlsa.get('phone','')}",
        f"  OLED L4: Bring: {dlsa.get('bring','')}",
        "└───────────────────────────────────┘",
    ]
    return "\n".join(lines)


def main():
    from fastapi import FastAPI, Request
    import uvicorn

    app = FastAPI(title="JusticeBridge UNO Q listener")

    @app.post("/signal")
    async def signal(request: Request):
        packet = await request.json()
        print("\n[UNO Q] signal received:")
        print(render(packet))
        return {"ok": True, "rendered": True}

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
