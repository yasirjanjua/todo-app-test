# 2048 Auto-Solver

A desktop app that watches a game of 2048 on screen and plays it for you — automatically,
regardless of whether the game is a website, an Electron app, a native binary, or an emulator,
and regardless of whether its tiles are numbers, fruit, or custom icons.

## How it works, in one paragraph

The app never touches the game's code, memory, or files. It takes a screenshot of the board
region roughly ten times a second, works out which tile is in which cell by comparing small
crops against pictures it learned earlier, runs those tiles through a game-tree search to pick
the best move, and then sends that move to the game exactly the way your keyboard would —
an arrow-key press. It re-reads the board after every move to confirm the press actually
worked before deciding what to do next. This is slower than reading the game's internal state
directly, but it is the only approach that works identically on every game, on every platform,
forever, without needing an update every time a game changes its code.

## Installing

Grab the build for your platform from the project's Releases page (or build it yourself —
see [Building from source](#building-from-source) below).

### macOS

1. Open the `.dmg` and drag **2048 Auto-Solver** into Applications.
2. Launch it. The app will ask for two permissions before it can do anything:
   - **Screen Recording** — lets it see your game, the same way a screenshot does.
   - **Accessibility** — lets it press arrow keys for you while it plays.

   For each one, click **Open Settings**, flip the toggle for 2048 Auto-Solver in the System
   Settings pane that opens, then come back to the app. It polls in the background and offers
   a **Relaunch now** button the moment it detects the permission was granted — macOS requires
   a fresh process launch for a new permission grant to take effect, so this step can't be
   skipped.
3. Click **Find my game**, pick your game window from the pictures shown, confirm the
   highlighted grid, and click **Start**.

> **Note on code signing:** the pre-built `.dmg` released from this project's CI is *not*
> notarized (that requires a paid Apple Developer account, which this project does not have).
> Gatekeeper will refuse to open it with a "damaged" or "unidentified developer" message. Right-
> click the app, choose **Open**, and confirm once — macOS remembers your choice after that. If
> you're building your own signed release, see the comment in
> `.github/workflows/build.yml`'s `build-macos` job for the exact `codesign` /
> `xcrun notarytool` commands to add once you have a Developer ID certificate.

### Windows

1. Download and unzip the release, then run `2048AutoSolver.exe`.
2. Click **Find my game**, pick your window, confirm the grid, click **Start**. There is no
   permission step on Windows.

> **Note on antivirus false positives:** Windows Defender (or your third-party antivirus) may
> flag or quarantine this app on first run. This is a **false positive**, not a sign of actual
> malware — but it's an extremely common one for this exact combination of tools: a
> PyInstaller-bundled Python executable that also sends synthetic keyboard input is a textbook
> heuristic match for keylogger/RAT malware, even though this app only ever sends the four
> arrow keys to whichever window you explicitly picked. If your antivirus blocks it, you'll
> need to allow it manually (usually via "More info -> Run anyway" in Defender's SmartScreen
> prompt, or an exclusion rule in your antivirus settings). A properly signed release (again,
> requiring a paid code-signing certificate this project does not currently have) would avoid
> most of this; see `.github/workflows/build.yml` for where a signing step would go.

### Linux (X11)

1. Make the AppImage executable and run it: `chmod +x 2048AutoSolver-x86_64.AppImage && ./2048AutoSolver-x86_64.AppImage`.
2. Click **Find my game**, pick your window, confirm the grid, click **Start**.

Wayland is not supported for capture or window enumeration in this release — run under Xorg,
or an XWayland session, if your desktop defaults to Wayland.

## Using it

1. **Find my game** — the app lists every visible window with a live thumbnail; click yours.
   (If you've played this exact window title before, this step is skipped entirely — the app
   remembers.)
2. **Confirm the grid** — a green rectangle shows where the app thinks the board is. Click
   **Looks right**, or **Let me adjust** to drag its corners.
3. **Teach me the tiles** — start a new game and click the button; the app learns each tile
   picture as it appears on screen, with a small "Learned a new tile" toast each time. No
   typing or labeling involved. If you're already mid-game, use the "I'm already mid-game"
   option instead.
4. **Start** — a small floating panel appears showing the board the app currently sees, its
   chosen move, search depth/decision time, and moves/sec, plus **Pause**, **Stop**, and
   **Save Snapshot** buttons. On Windows and Linux these also work as global hotkeys even
   while the game window has focus:
   - `P` — pause / resume
   - `Q` — stop (always releases any held key, even mid-move)
   - `S` — save a debug snapshot of the current board crop, for troubleshooting a misread

   **On macOS, global hotkeys are off by default** (see the note below) — use the HUD's
   buttons instead, which work identically and don't need focus on any particular window.

Recalibrate any time from the "ready to play" screen if you change your browser zoom, theme,
or window size.

> **Note on macOS global hotkeys:** the underlying library (`pynput`) resolves each hotkey
> character through a macOS keyboard-layout API that, on some macOS versions, asserts it's
> only ever called from the main thread and hard-crashes the whole app otherwise -- a crash no
> amount of Python error handling can catch, since it's an OS-level abort, not a Python
> exception. Rather than risk that, global hotkeys are disabled by default on macOS; the HUD's
> Pause/Stop/Save Snapshot buttons cover the same functionality without touching the affected
> code path at all. If you want to try enabling them anyway, set
> `AppConfig.enable_macos_global_hotkeys = True` in your own launch script -- expect a possible
> crash on some macOS versions.

## Troubleshooting

**The grid overlay doesn't line up with the board.** Click **Let me adjust** on the grid
confirmation screen and drag the corner handles (or drag the middle of the box to move the
whole thing). If auto-detection is consistently off for your particular game (e.g. it has an
unusually thin or thick border), the app falls back to detecting the outer board container —
usually still close enough for a manual nudge — rather than leaving the ROI where it was.

If the app couldn't confidently detect anything (the whole window shows behind a green box
covering almost all of it, with a warning above it), it opens straight into adjust mode and
*requires* you to drag the box down to just the 4x4 grid before continuing — don't click
**Looks right** without actually checking in this case. Proceeding with an oversized box means
every "tile" the app learns afterward is really just random webpage content around the board,
which shows up as implausible values in the HUD (see the next entry) and gets nowhere.

**The app pauses with "Saw a tile I can't place even after trying to learn it."** A new tile
tier the recognizer has never seen is normally learned automatically and silently mid-play —
since tiers above 2 can only ever appear via a merge, a never-seen sprite is unambiguous and
gets added on the spot, no pause needed. This message means it genuinely couldn't be resolved
that way (most commonly: calibration was finished before the tier-2 confirmation step ever
came up). Press `S` to save the offending crop for inspection, then use **Recalibrate** and
re-run tile learning.

**The HUD shows huge, implausible tile values (e.g. 2048, 32768) on a game that just
started, or pauses saying "This doesn't look like the game board anymore."** This means
recognition itself has gone unreliable, almost always because the calibrated region is no
longer actually looking at the board — most commonly a page reflow (an ad loading, a layout
shift) moved the board within an otherwise-unmoved window, which the window-level "did it
move" check can't catch since the window itself didn't move. Each poll then sees different
content and mistakes it for a stream of brand-new tiles. The app caps how many new tiles a
single reading can plausibly learn and pauses instead of guessing further the moment that cap
is exceeded, and will *not* save those bogus tiles into your profile — but the fix is always
the same: **Stop**, then **Recalibrate** to re-align the grid.

**The app plays moves that don't do anything, or seems "stuck."** Every move is verified by
re-reading the board and confirming it changed; an unchanged board is automatically treated as
illegal and the app tries the next-best move instead of repeating the same keystroke. If *all
four* directions produce no change, the app concludes the game is over. If this triggers
incorrectly (e.g. a game with an unusual animation), try raising the "settle-frame count" or
lowering key-hold duration in the Advanced panel.

**The window moved and now the app pauses.** This is deliberate (Part 5 of the design spec):
rather than silently reading the wrong pixels after a window is moved or resized, the app
detects the mismatch and pauses with a clear message. Move the window back, or run
**Recalibrate**.

## Advanced panel

Collapsed by default; expand it for search depth, per-heuristic weights, the recognition
confidence threshold, settle-frame count, and **dry-run mode** — the primary debugging tool,
which runs the full perceive/decide pipeline and logs its decisions without ever pressing a
key.

## Building from source

Use **Python 3.11 or 3.12** specifically, not whatever the newest release on your system is.
Every pin in `requirements.txt` was verified against that range; several of them (PySide6,
opencv, and especially the macOS `pyobjc-framework-*` packages) don't yet publish prebuilt
wheels for brand-new Python releases (3.13/3.14 at time of writing), which makes `pip` fall
back to building from source -- and older sdists like `pyobjc-framework-Quartz`'s frequently
fail that build with a `pkg_resources`/`setuptools` incompatibility that has nothing to do
with this project. If you hit that error, it means your venv's Python is too new; recreate it
with 3.11/3.12 rather than trying to patch around the build failure.

```bash
cd 2048-auto-solver
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Run the test suite (pure-Python, no screen or OS access required):

```bash
python -m pytest tests/ -v
```

Build a native package for your current OS (PyInstaller cannot cross-compile — build each OS
on that OS):

```bash
pyinstaller packaging/pyinstaller.spec --noconfirm --clean
```

See `.github/workflows/build.yml` for the full CI matrix (macOS `.dmg`, Windows `.zip`
portable build, Linux `.AppImage`) and the packaging notes on code signing and antivirus.

## Project layout

```
core/       Pure game logic: bitboard, heuristics, expectimax solver. No I/O of any kind.
vision/     Perception: grid detection, tile recognition, tile learning, frame-stability.
backends/   Per-OS capture (mss/dxcam) and input (pynput/pydirectinput) implementations,
            hidden behind OS-agnostic protocols. All platform-specific code lives here.
app/        Wires the above together: profiles, the wizard state machine, the play loop.
ui/         PySide6 wizard, play HUD, global hotkeys, advanced settings panel.
tests/      Unit tests for core/ (fixed boards) plus an integration smoke test for the
            closed-loop play controller against a fake in-memory game.
```

## Design assumptions worth knowing

- **Perception is pixels-only, action is keystrokes-only.** No debugger attachment, no
  process-memory reads, no file patching, no page injection. Slower and less precise than
  those approaches, but it's the only one that works on every game, forever, without
  per-game maintenance.
- **Tile *rank*, never tile *value*.** Recognition maps a sprite to an ordinal tier; tier `n`
  is treated internally as `2**n`. This is what makes a fruit-tile game and a numeric game the
  same code path.
- **Moves are capped around 6–8/second by the game's own animation**, not by anything this app
  could speed up — capture takes 1–5ms, recognition 5–15ms, search 20–200ms, and the
  slide/merge animation 100–150ms dominates every move. Frame-stability polling (wait for N
  consecutive near-identical frames) is what makes the app robust to that animation instead of
  guessing a fixed delay.
