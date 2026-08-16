"""PySide6 (Qt) interface: the wizard, the play HUD, and global hotkeys.

Nothing outside this package should import ``PySide6`` -- ``app/`` and everything below it
stays UI-framework-agnostic so the wizard state machine and play loop can be driven by a
different frontend later without touching them. Rationale for PySide6 itself (mature,
cross-platform, supports the frameless always-on-top HUD) is documented in the project
README; the ~60MB bundle-size cost is accepted deliberately.
"""
