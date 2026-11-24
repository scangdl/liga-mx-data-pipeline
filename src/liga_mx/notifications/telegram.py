#!/usr/bin/env python3
"""
Telegram Notifier — Envía alertas y resúmenes a Telegram.
"""

from __future__ import annotations

import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

from liga_mx.config import get_config, load_config


class TelegramNotifier:
    """Cliente simple para enviar mensajes a Telegram."""
    
    def __init__(self):
        load_config()
        self.cfg = get_config()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or self.cfg.telegram.bot_token
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID") or self.cfg.telegram.chat_id
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            print("[Telegram] WARNING: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados")
    
    def send(self, text: str, parse_mode: Optional[str] = "HTML") -> bool:
            """Envía mensaje a Telegram."""
            if not self.enabled:
                return False

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    return True
                else:
                    print(f"[Telegram] Error {resp.status_code}: {resp.text}")
                    return False
            except Exception as e:
                print(f"[Telegram] Exception: {e}")
                return False
    
    def send_cron_summary(self, job_name: str, status: str, details: dict, duration_sec: float) -> bool:
        """Envía resumen de ejecución de cron job."""
        emoji = "✅" if status == "ok" else "❌"
        
        lines = [
            f"{emoji} <b>{job_name}</b> — {status.upper()}",
            f"⏱ {duration_sec:.1f}s | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if "btts" in details:
            b = details["btts"]
            lines.append(f"\n📊 <b>BTTS</b>: AUC={b.get('auc',0):.3f} Brier={b.get('brier',0):.3f} Acc={b.get('accuracy',0):.1%}")
        if "over25" in details:
            o = details["over25"]
            lines.append(f"📊 <b>Over2.5</b>: AUC={o.get('auc',0):.3f} Brier={o.get('brier',0):.3f} Acc={o.get('accuracy',0):.1%}")
        if "one_x_two" in details:
            x = details["one_x_two"]
            lines.append(f"📊 <b>1X2</b>: Acc={x.get('accuracy',0):.1%}")
        if "n_test" in details:
            lines.append(f"📈 Partidos test: {details['n_test']}")
        
        if status != "ok":
            lines.append(f"\n⚠️ Error: {details.get('error', 'Unknown')}")
        
        return self.send("\n".join(lines))
    
    def send_decay_alert(self, alert_data: dict) -> bool:
        """Envía alerta de model decay."""
        lines = [
            "🔴 <b>ALERTA: Model Decay Detectado</b>",
            f"Modelo: {alert_data.get('model', 'N/A')}",
            f"Periodo: {alert_data.get('period', 'N/A')}",
            f"Partidos: {alert_data.get('matches', 'N/A')}",
        ]
        
        for key, val in alert_data.items():
            if key in ["brier", "auc", "calibration", "btts_rate_shift"]:
                lines.append(f"  {key}: {val}")
        
        return self.send("\n".join(lines))
    
    def send_backtest_summary(self, results: dict) -> bool:
        """Envía resumen de backtest."""
        lines = [
            f"📊 <b>Backtest: {results.get('season', 'N/A')}</b>",
            f"Periodo: {results.get('test_period', 'N/A')}",
            f"Train: {results.get('n_train', 0)} | Test: {results.get('n_test', 0)}",
            "",
            f"BTTS: AUC={results.get('btts',{}).get('auc',0):.3f} Brier={results.get('btts',{}).get('brier',0):.3f} Acc={results.get('btts',{}).get('accuracy',0):.1%}",
            f"Over2.5: AUC={results.get('over25',{}).get('auc',0):.3f} Brier={results.get('over25',{}).get('brier',0):.3f} Acc={results.get('over25',{}).get('accuracy',0):.1%}",
            f"1X2: Acc={results.get('one_x_two',{}).get('accuracy',0):.1%}",
        ]
        return self.send("\n".join(lines))
    
    def send_paper_trade_update(self, market: str, action: str, bets: list, bankroll: float) -> bool:
        """Envía actualización de paper trading."""
        if not bets:
            return self.send(f"📝 <b>Paper Trading {market}</b>: {action} — Sin apuestas nuevas")
        
        lines = [f"📝 <b>Paper Trading {market}</b>: {action}"]
        lines.append(f"💰 Bankroll: {bankroll:.2f} | Apuestas: {len(bets)}")
        
        for b in bets[:5]:  # máx 5
            lines.append(f"  {b.get('home','?')} vs {b.get('away','?')} | {b.get('bet','?')} @ {b.get('odds',0):.2f} | Stake: {b.get('stake',0):.2f}")
        
        if len(bets) > 5:
            lines.append(f"  ... y {len(bets)-5} más")
        
        return self.send("\n".join(lines))


# Instancia global
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def notify(text: str) -> bool:
    """Función helper para envío rápido."""
    return get_notifier().send(text)


if __name__ == "__main__":
    # Test rápido
    notifier = TelegramNotifier()
    if notifier.enabled:
        notifier.send("🤖 Liga MX Bot conectado correctamente")
    else:
        print("Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env")