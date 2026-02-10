"""
Custom styled widgets for the Soccer Predictions App.
Dark-themed Tkinter widgets with modern look.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import GUI_SETTINGS

C = GUI_SETTINGS["colors"]


class StyledFrame(tk.Frame):
    """Dark themed frame."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C["bg_dark"])
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, **kwargs)


class CardFrame(tk.Frame):
    """Card-style frame with rounded appearance."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C["bg_card"])
        kwargs.setdefault("highlightbackground", C["border"])
        kwargs.setdefault("highlightthickness", 1)
        kwargs.setdefault("padx", 15)
        kwargs.setdefault("pady", 10)
        super().__init__(parent, **kwargs)


class StyledLabel(tk.Label):
    """Dark themed label."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg") if hasattr(parent, "cget") else C["bg_dark"])
        kwargs.setdefault("fg", C["text_primary"])
        kwargs.setdefault("font", ("Segoe UI", 11))
        super().__init__(parent, **kwargs)


class HeaderLabel(tk.Label):
    """Large header label."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg") if hasattr(parent, "cget") else C["bg_dark"])
        kwargs.setdefault("fg", C["accent"])
        kwargs.setdefault("font", ("Segoe UI", 18, "bold"))
        super().__init__(parent, **kwargs)


class SubHeaderLabel(tk.Label):
    """Sub-header label."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg") if hasattr(parent, "cget") else C["bg_dark"])
        kwargs.setdefault("fg", C["text_primary"])
        kwargs.setdefault("font", ("Segoe UI", 14, "bold"))
        super().__init__(parent, **kwargs)


class MutedLabel(tk.Label):
    """Muted/secondary text label."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg") if hasattr(parent, "cget") else C["bg_dark"])
        kwargs.setdefault("fg", C["text_muted"])
        kwargs.setdefault("font", ("Segoe UI", 9))
        super().__init__(parent, **kwargs)


class AccentButton(tk.Button):
    """Styled accent button."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C["accent"])
        kwargs.setdefault("fg", C["bg_dark"])
        kwargs.setdefault("activebackground", "#00a8cc")
        kwargs.setdefault("activeforeground", C["bg_dark"])
        kwargs.setdefault("font", ("Segoe UI", 11, "bold"))
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("cursor", "hand2")
        kwargs.setdefault("padx", 20)
        kwargs.setdefault("pady", 8)
        kwargs.setdefault("borderwidth", 0)
        super().__init__(parent, **kwargs)
        self.bind("<Enter>", lambda e: self.config(bg="#00a8cc"))
        self.bind("<Leave>", lambda e: self.config(bg=C["accent"]))


class SecondaryButton(tk.Button):
    """Styled secondary button."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C["bg_light"])
        kwargs.setdefault("fg", C["text_primary"])
        kwargs.setdefault("activebackground", C["bg_card"])
        kwargs.setdefault("activeforeground", C["text_primary"])
        kwargs.setdefault("font", ("Segoe UI", 10))
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("cursor", "hand2")
        kwargs.setdefault("padx", 15)
        kwargs.setdefault("pady", 6)
        kwargs.setdefault("borderwidth", 0)
        super().__init__(parent, **kwargs)
        self.bind("<Enter>", lambda e: self.config(bg=C["bg_card"]))
        self.bind("<Leave>", lambda e: self.config(bg=C["bg_light"]))


class NavButton(tk.Button):
    """Navigation sidebar button."""
    def __init__(self, parent, **kwargs):
        self._active = kwargs.pop("active", False)
        kwargs.setdefault("bg", C["bg_medium"] if not self._active else C["accent"])
        kwargs.setdefault("fg", C["text_primary"] if not self._active else C["bg_dark"])
        kwargs.setdefault("activebackground", C["accent"])
        kwargs.setdefault("activeforeground", C["bg_dark"])
        kwargs.setdefault("font", ("Segoe UI", 11))
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("cursor", "hand2")
        kwargs.setdefault("anchor", "w")
        kwargs.setdefault("padx", 20)
        kwargs.setdefault("pady", 12)
        kwargs.setdefault("borderwidth", 0)
        super().__init__(parent, **kwargs)
        if not self._active:
            self.bind("<Enter>", lambda e: self.config(bg=C["bg_light"]))
            self.bind("<Leave>", lambda e: self.config(bg=C["bg_medium"]))

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.config(bg=C["accent"], fg=C["bg_dark"])
        else:
            self.config(bg=C["bg_medium"], fg=C["text_primary"])


class ProbabilityBar(tk.Canvas):
    """Horizontal probability bar showing Home/Draw/Away probabilities."""
    def __init__(self, parent, home_prob: float = 0.33, draw_prob: float = 0.33,
                 away_prob: float = 0.34, width: int = 300, height: int = 28, **kwargs):
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, width=width, height=height, **kwargs)
        self.config(bg=parent.cget("bg") if hasattr(parent, "cget") else C["bg_dark"])
        self._width = width
        self._height = height
        self.draw_bar(home_prob, draw_prob, away_prob)

    def draw_bar(self, home_prob: float, draw_prob: float, away_prob: float):
        self.delete("all")
        w = self._width
        h = self._height

        # Ensure probabilities sum to 1
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total

        # Draw segments
        x = 0
        hw = int(w * home_prob)
        dw = int(w * draw_prob)
        aw = w - hw - dw

        # Home (green)
        if hw > 0:
            self.create_rectangle(x, 2, x + hw - 1, h - 2, fill=C["win_color"], outline="")
            if hw > 30:
                self.create_text(x + hw // 2, h // 2, text=f"{home_prob:.0%}",
                               fill=C["bg_dark"], font=("Segoe UI", 8, "bold"))
        x += hw

        # Draw (yellow)
        if dw > 0:
            self.create_rectangle(x, 2, x + dw - 1, h - 2, fill=C["draw_color"], outline="")
            if dw > 30:
                self.create_text(x + dw // 2, h // 2, text=f"{draw_prob:.0%}",
                               fill=C["bg_dark"], font=("Segoe UI", 8, "bold"))
        x += dw

        # Away (red)
        if aw > 0:
            self.create_rectangle(x, 2, x + aw, h - 2, fill=C["lose_color"], outline="")
            if aw > 30:
                self.create_text(x + aw // 2, h // 2, text=f"{away_prob:.0%}",
                               fill=C["bg_dark"], font=("Segoe UI", 8, "bold"))


class ConfidenceMeter(tk.Canvas):
    """Circular confidence meter."""
    def __init__(self, parent, confidence: float = 0.5, size: int = 60, **kwargs):
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, width=size, height=size, **kwargs)
        self.config(bg=parent.cget("bg") if hasattr(parent, "cget") else C["bg_dark"])
        self._size = size
        self.draw_meter(confidence)

    def draw_meter(self, confidence: float):
        self.delete("all")
        s = self._size
        pad = 4
        extent = int(confidence * 360)

        # Color based on confidence
        if confidence >= 0.6:
            color = C["accent_green"]
        elif confidence >= 0.4:
            color = C["accent_yellow"]
        else:
            color = C["accent_red"]

        # Background arc
        self.create_arc(pad, pad, s - pad, s - pad, start=90, extent=360,
                       outline=C["border"], width=4, style="arc")

        # Confidence arc
        self.create_arc(pad, pad, s - pad, s - pad, start=90, extent=-extent,
                       outline=color, width=4, style="arc")

        # Text
        self.create_text(s // 2, s // 2, text=f"{confidence:.0%}",
                        fill=color, font=("Segoe UI", 10, "bold"))


class ScrollableFrame(tk.Frame):
    """Scrollable frame container with macOS trackpad + mousewheel support."""
    def __init__(self, parent, **kwargs):
        bg = kwargs.pop("bg", C["bg_dark"])
        super().__init__(parent, bg=bg)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=bg)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel — works on macOS, Windows, and Linux
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # Make scrollable frame expand to canvas width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        import platform
        system = platform.system()
        if system == "Darwin":
            # macOS — trackpad and mouse wheel
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_mac)
        elif system == "Windows":
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_win)
        else:
            # Linux
            self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel_mac(self, event):
        """macOS: event.delta is small integers (e.g. -1, 1, -3, 3 for trackpad)."""
        self.canvas.yview_scroll(int(-1 * event.delta), "units")

    def _on_mousewheel_win(self, event):
        """Windows: event.delta is typically 120 or -120."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        """Linux: Button-4 = scroll up, Button-5 = scroll down."""
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class StatusBar(tk.Frame):
    """Bottom status bar."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["bg_medium"], height=30)
        self.pack_propagate(False)

        self.status_label = tk.Label(self, text="Ready", bg=C["bg_medium"],
                                     fg=C["text_muted"], font=("Segoe UI", 9),
                                     anchor="w")
        self.status_label.pack(side="left", padx=10, fill="x", expand=True)

        self.info_label = tk.Label(self, text="", bg=C["bg_medium"],
                                    fg=C["text_muted"], font=("Segoe UI", 9),
                                    anchor="e")
        self.info_label.pack(side="right", padx=10)

    def set_status(self, text: str):
        self.status_label.config(text=text)

    def set_info(self, text: str):
        self.info_label.config(text=text)
