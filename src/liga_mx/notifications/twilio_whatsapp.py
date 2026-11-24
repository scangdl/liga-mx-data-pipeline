#!/usr/bin/env python3
"""
Twilio WhatsApp Notifier — Envía alertas a WhatsApp via Twilio.

Setup:
1. Cuenta Twilio: https://www.twilio.com/try-twilio
2. WhatsApp Sandbox: Console → Messaging → Try it out → WhatsApp
3. Une número a sandbox (join <code> al +14155238886)
4. Credenciales en .env:
   TWILIO_ACCOUNT_SID=ACxxxx
   TWILIO_AUTH_TOKEN=xxxx
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886  # sandbox number
   TWILIO_WHATSAPP_TO=whatsapp:+52xxxxxxxxxx   # tu número verificado
"""
from __future__ import annotations

import os
from typing import Optional

from liga_mx.config import load_config


class TwilioWhatsAppNotifier:
    """Cliente para enviar mensajes a WhatsApp via Twilio."""

    def __init__(self):
        load_config()
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
        self.to_number = os.getenv("TWILIO_WHATSAPP_TO")
        self.enabled = bool(self.account_sid and self.auth_token and self.to_number)

        if not self.enabled:
            missing = []
            if not self.account_sid: missing.append("TWILIO_ACCOUNT_SID")
            if not self.auth_token: missing.append("TWILIO_AUTH_TOKEN")
            if not self.to_number: missing.append("TWILIO_WHATSAPP_TO")
            print(f"[Twilio WhatsApp] WARNING: faltan {', '.join(missing)} en .env")

    def send(self, text: str) -> bool:
        """Envía mensaje a WhatsApp."""
        if not self.enabled:
            return False

        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)

            message = client.messages.create(
                body=text,
                from_=self.from_number,
                to=self.to_number
            )
            print(f"[Twilio WhatsApp] Enviado SID: {message.sid}")
            return True
        except Exception as e:
            print(f"[Twilio WhatsApp] Error: {e}")
            return False

    def send_alert(self, title: str, details: dict) -> bool:
        """Envía alerta estructurada."""
        lines = [f"🔔 {title}"]
        for k, v in details.items():
            lines.append(f"  {k}: {v}")
        return self.send("\n".join(lines))


# Instancia global
_notifier: Optional[TwilioWhatsAppNotifier] = None


def get_twilio_notifier() -> TwilioWhatsAppNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TwilioWhatsAppNotifier()
    return _notifier


def notify_whatsapp(text: str) -> bool:
    """Helper para envío rápido."""
    return get_twilio_notifier().send(text)


if __name__ == "__main__":
    notifier = TwilioWhatsAppNotifier()
    if notifier.enabled:
        notifier.send("🤖 Liga MX Bot conectado a WhatsApp (Twilio)")
    else:
        print("Configura TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_TO en .env")