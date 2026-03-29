from __future__ import annotations

import time

import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.selfdrive.ui.body.widgets.pairing_dialog import BodyPairingScreen
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.body.animations import (
  FaceAnimator,
  ASLEEP,
  FACE_EYES_OPEN_BROWS,
  INQUISITIVE,
  LISTENING,
  MOUTH_TALK_FRAMES,
  NORMAL,
  SLEEPY,
)
from opendbc.car.body.values import CAR

GRID_COLS = 16
GRID_ROWS = 8
RADIUS = 50 if gui_app.big_ui() else 10

IDLE_TIMEOUT = 30.0        # seconds of no joystick input before playing INQUISITIVE
IDLE_STEER_THRESH = 0.5    # degrees — below this counts as no input
IDLE_SPEED_THRESH = 0.01   # m/s — below this counts as no input

PAIR_BTN_FONT_SIZE = 60
PAIR_BTN_MARGIN = 20


class BodyLayout(Widget):
  def __init__(self):
    super().__init__()
    self._animator = FaceAnimator(ASLEEP)
    self._turning_left = False
    self._turning_right = False
    self._last_input_time = time.monotonic()
    self._was_active = False
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._prev_joystick_debug_mode = False

    self.pairing_button = Button("CONNECT", font_size=PAIR_BTN_FONT_SIZE, font_weight=FontWeight.BOLD,
                        click_callback=lambda: gui_app.push_widget(BodyPairingScreen()),
                        button_style=ButtonStyle.ACTION)

  def set_settings_callback(self, callback):
    pass

  def is_swiping_left(self) -> bool:
    return False

  def draw_dot_grid(self, rect: rl.Rectangle, dots: list[tuple[int, int]], color: rl.Color | None = None):
    if color is None:
      color = rl.WHITE

    spacing = (rect.height) / (GRID_ROWS)

    grid_w = (GRID_COLS - 1) * spacing
    grid_h = (GRID_ROWS - 1) * spacing

    offset_x = rect.x + (rect.width - grid_w) / 2
    offset_y = rect.y + (rect.height - grid_h) / 2

    for row, col in dots:
      x = int(offset_x + col * spacing)
      y = int(offset_y + row * spacing)
      rl.draw_circle(x, y, RADIUS, color)

  def _draw_pair_button(self, rect: rl.Rectangle):
    text = tr(tr_noop("CONNECT"))
    text_size = measure_text_cached(self._font_bold, text, PAIR_BTN_FONT_SIZE)
    btn_w = int(text_size.x + 200)
    btn_h = 200
    btn_x = int(rect.x + rect.width - btn_w - PAIR_BTN_MARGIN)
    btn_y = int(rect.y + rect.height - btn_h - PAIR_BTN_MARGIN)

    self.pairing_button.render(rl.Rectangle(btn_x, btn_y, btn_w, btn_h))

  def _update_state(self):
    sm = ui_state.sm

    body_interactive = ui_state.joystick_debug_mode or ui_state.body_voice_session
    active = ui_state.is_onroad() or ui_state.body_voice_session
    if active and body_interactive:
      if not self._was_active:
        self._last_input_time = time.monotonic()
        self._was_active = True

      cs = sm['carState']
      has_input = abs(cs.steeringAngleDeg) > IDLE_STEER_THRESH or abs(cs.vEgo) > IDLE_SPEED_THRESH
      if has_input:
        self._last_input_time = time.monotonic()

      if ui_state.body_assistant_listening:
        self._animator.set_animation(LISTENING)
      elif time.monotonic() - self._last_input_time > IDLE_TIMEOUT:
        self._animator.set_animation(INQUISITIVE)
      else:
        self._animator.set_animation(NORMAL)
    else:
      self._was_active = False
      self._animator.set_animation(ASLEEP)

    if not sm.updated['carState']:
      return

    steer = sm['testJoystick'].axes[1] if len(sm['testJoystick'].axes) > 1 else 0
    is_v2 = sm['carParams'].carFingerprint == CAR.COMMA_BODY_V2
    if is_v2:
      self._turning_left = steer <= -0.05
      self._turning_right = steer >= 0.05
    else:
      self._turning_left = steer >= 0.05
      self._turning_right = steer <= -0.05

  # play animation on screen tap
  def _handle_mouse_release(self, mouse_pos):
    if gui_app.big_ui():
      # allow pairing button to work
      pair_rect = self.pairing_button.rect
      if rl.check_collision_point_rec(mouse_pos, pair_rect):
        self.pairing_button._click_callback()
        return

    super()._handle_mouse_release(mouse_pos)
    if not self._was_active:
      self._animator.set_animation(SLEEPY)

  def _render(self, rect: rl.Rectangle):
    dots = self._animator.get_dots()
    color = None
    if ui_state.body_assistant_speaking:
      phase = int(time.monotonic() * 10.0) % len(MOUTH_TALK_FRAMES)
      dots = list(FACE_EYES_OPEN_BROWS) + list(MOUTH_TALK_FRAMES[phase])
    elif ui_state.body_assistant_listening:
      color = rl.Color(100, 200, 255, 255)
    animation = self._animator._animation
    if self._turning_left and animation.left_turn_remove:
      remove_set = set(animation.left_turn_remove)
      dots = [d for d in dots if d not in remove_set]
    elif self._turning_right and animation.right_turn_remove:
      remove_set = set(animation.right_turn_remove)
      dots = [d for d in dots if d not in remove_set]
    self.draw_dot_grid(rect, dots, color)
    if gui_app.big_ui():
      if ui_state.joystick_debug_mode:
        for widget in gui_app._nav_stack:
          if isinstance(widget, BodyPairingScreen):
            gui_app.pop_widget()
            break
      else:
        self._draw_pair_button(rect)
